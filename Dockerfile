# cu-lsh: CUDA Locality Sensitive Hashing
# Build: docker build -t culsh .
# Run:   docker run --gpus all -it culsh

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    gpg \
    wget \
    git \
    python3 \
    python3-pip \
    python3-dev \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY python/requirements.txt python/requirements.txt
COPY python/requirements_dev.txt python/requirements_dev.txt
RUN pip install --no-cache-dir pybind11 -r python/requirements.txt

COPY cuda/ cuda/
COPY python/ python/

WORKDIR /app/python
RUN pip install --no-cache-dir --no-build-isolation -e .

WORKDIR /app

CMD ["/bin/bash"]
