FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install basic utilities and Miniconda
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    vim \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

ENV PATH=$CONDA_DIR/bin:$PATH

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create Conda environment
RUN conda env create -f environment.yml && conda clean -afy

# Activate Conda environment
# We set this in ~/.bashrc so it activates on login, and also set PATH
RUN echo "source activate marconi" > ~/.bashrc
ENV PATH /opt/conda/envs/marconi/bin:$PATH

# Default command
CMD ["/bin/bash"]