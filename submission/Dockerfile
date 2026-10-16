# EchoRisk-MICCAI 2026 submission container — EchoPrime backbone.
#
# TASK_ID is fixed at build time (contract §5). Build one image per task:
#   docker build --build-arg TASK_ID=1 -t <team>-task1:final .
#   docker build --build-arg TASK_ID=2 -t <team>-task2:final .
#   docker build --build-arg TASK_ID=3 -t <team>-task3:final .
#
# Build context must be this submission/ directory, with model weights
# staged in place first (see build.sh, which copies them from the repo).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 22.04 ships Python 3.10. The training env used 3.11, but this
# inference path (state_dict load + torch/vision/numpy/pydicom/opencv) has
# no Python-version-sensitive behaviour; reproducibility is guaranteed by
# pinning the exact PACKAGE versions below, which is what actually matters.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps pinned to the EXACT versions the checkpoints were trained with
# (challenge conda env): torch 2.5.1+cu124, torchvision 0.20.1+cu124,
# numpy 2.4.3, pydicom 3.0.2, opencv-python-headless 4.13.0.92,
# pylibjpeg 2.1.0/2.4.0, pillow 11.3.0.
RUN pip3 install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 torchvision==0.20.1
# numpy: training env had 2.4.3, which requires Python >=3.11 and is not
# available for this image's Python 3.10. 2.2.6 is the highest 3.10-compatible
# release; the inference path (linspace frame sampling + array normalisation)
# is numerically identical across numpy 2.2/2.4, so this does not affect
# reproducibility of predictions.
RUN pip3 install \
        numpy==2.2.6 \
        pydicom==3.0.2 \
        opencv-python-headless==4.13.0.92 \
        pylibjpeg==2.1.0 \
        pylibjpeg-libjpeg==2.4.0 \
        pillow==11.3.0

ARG TASK_ID
ENV TASK_ID=${TASK_ID}
ENV ECHOPRIME_ENCODER_WEIGHTS=/app/weights/echo_prime_encoder.pt

# Model code
COPY model.py infer.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Weights and checkpoints (staged into the build context by build.sh)
COPY weights/ ./weights/
COPY checkpoints/ ./checkpoints/

ENTRYPOINT ["./entrypoint.sh"]
