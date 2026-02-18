# ==============================================================================
# Dockerfile — mlsys-marconi
#
# Build for GPU-backed inference.
#
# Marconi CPU-based trace repro is run outside Docker per marconi/artifact_evaluation.md
# (conda env + run_all_experiments.sh).
#
# Build:  docker compose build sglang-server
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage: base — CUDA 12.4 + Ubuntu 22.04 + common tooling
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ------------------------------------------------------------------------------
# Target: sglang — inference server + benchmarking (GPU)
# Versions pinned in this file; uses uv for Python/sglang install.
# ------------------------------------------------------------------------------
FROM base AS sglang

# -- uv -----------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# -- Pinned versions -----------------------------------------------------------
ENV PYTHON_VERSION=3.11 \
    SGLANG_VERSION=0.5.6.post2 \
    CUDA_SHORT=cu124 \
    TORCH_VERSION=2.5

# -- Python + venv -------------------------------------------------------------
RUN uv python install ${PYTHON_VERSION}
RUN uv venv /opt/venv --python ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

# -- Install SGLang from PyPI (minimal; see docs.sglang.io/get_started/install.html)
RUN uv pip install "sglang==${SGLANG_VERSION}" \
    --find-links "https://flashinfer.ai/whl/${CUDA_SHORT}/torch${TORCH_VERSION}/flashinfer-python"

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
