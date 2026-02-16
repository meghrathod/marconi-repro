# ==============================================================================
# Dockerfile — mlsys-marconi
#
# Multi-target build for Chameleon Cloud GPU instances (A100).
#
# Targets
# -------
#   marconi-sim : Marconi trace-driven simulation (CPU-only, conda env)
#   sglang      : SGLang inference server + benchmark tooling (GPU, uv)
#
# Build examples
# --------------
#   docker compose build marconi-sim
#   docker compose build sglang-server
#
#   # or standalone:
#   docker build --target marconi-sim -t marconi-sim .
#   docker build --target sglang      -t marconi-sglang \
#       --build-arg SGLANG_BRANCH=support_mamba_radix_cache .
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage: base — CUDA 12.4 + Ubuntu 22.04 + common tooling
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        jq \
        lsb-release \
        software-properties-common \
        vim \
        wget \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ------------------------------------------------------------------------------
# Target: marconi-sim — trace-driven simulation (CPU-only)
#
# Recreates the upstream marconi conda env.  Suitable for running the
# experiments in marconi/run_all_experiments.sh and the unit tests in tests/.
# ------------------------------------------------------------------------------
FROM base AS marconi-sim

# -- Miniconda -----------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/bin:${PATH}

# -- Conda environment from upstream spec -------------------------------------
COPY marconi/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy && \
    rm /tmp/environment.yml

# Activate marconi env by default
ENV PATH=/opt/conda/envs/marconi/bin:${PATH}
SHELL ["conda", "run", "--no-capture-output", "-n", "marconi", "/bin/bash", "-c"]

# -- Copy simulation code & tests ---------------------------------------------
COPY marconi/ ./marconi/
COPY tests/   ./tests/

CMD ["conda", "run", "--no-capture-output", "-n", "marconi", "bash"]

# ------------------------------------------------------------------------------
# Target: sglang — inference server + benchmarking (GPU)
#
# Uses uv for fast, reproducible Python environment management.
# By default installs SGLang from PyPI; set SGLANG_BRANCH to build from a
# specific git branch (e.g. support_mamba_radix_cache for PR #11214).
# ------------------------------------------------------------------------------
FROM base AS sglang

# -- uv -----------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# -- Configurable build args --------------------------------------------------
ARG PYTHON_VERSION=3.11
ARG SGLANG_VERSION=""
ARG SGLANG_BRANCH=""
ARG CUDA_SHORT=cu124
ARG TORCH_VERSION=2.5

# -- Python + venv -------------------------------------------------------------
RUN uv python install ${PYTHON_VERSION}
RUN uv venv /opt/venv --python ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# -- Install SGLang ------------------------------------------------------------
# Priority: SGLANG_VERSION (PyPI release) > SGLANG_BRANCH (git) > latest PyPI
RUN set -ex; \
    FL="https://flashinfer.ai/whl/${CUDA_SHORT}/torch${TORCH_VERSION}/flashinfer-python"; \
    if [ -n "${SGLANG_VERSION}" ]; then \
        echo "Installing SGLang ${SGLANG_VERSION} from PyPI"; \
        uv pip install "sglang[all]==${SGLANG_VERSION}" --find-links "${FL}"; \
    elif [ -n "${SGLANG_BRANCH}" ]; then \
        echo "Installing SGLang from branch ${SGLANG_BRANCH}"; \
        git clone --depth 1 -b "${SGLANG_BRANCH}" \
            https://github.com/sgl-project/sglang.git /opt/sglang-src; \
        uv pip install -e "/opt/sglang-src/python[all]" --find-links "${FL}"; \
    else \
        echo "Installing latest SGLang from PyPI"; \
        uv pip install "sglang[all]" --find-links "${FL}"; \
    fi

# -- Extra runtime deps -------------------------------------------------------
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN uv pip install -r /tmp/requirements-docker.txt && rm /tmp/requirements-docker.txt

# -- Copy project code ---------------------------------------------------------
COPY . .

# -- HuggingFace model cache (mount as a volume for persistence) ---------------
ENV HF_HOME=/workspace/.cache/huggingface
VOLUME ["/workspace/.cache/huggingface"]

EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:30000/health || exit 1

CMD ["python3", "-m", "sglang.launch_server", \
     "--host", "0.0.0.0", "--port", "30000"]
