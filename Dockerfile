FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libffi-dev \
    libssl-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

COPY env_laplace.yml /tmp/env_laplace.yml

RUN conda env create -f /tmp/env_laplace.yml && \
    conda clean -afy

ENV PATH /opt/conda/envs/openood_laplace/bin:$PATH
ENV CONDA_DEFAULT_ENV openood_laplace

RUN echo "source activate openood_laplace" > ~/.bashrc
