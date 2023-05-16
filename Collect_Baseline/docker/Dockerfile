FROM python:3.10-slim

LABEL maintainer="Laura Boyette <laura.boyette@stud.tu-darmstadt.de>" \
      desc="Docker image to run training on IF dataset"

# Install base dependencies, python packages and configure environment
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3-virtualenv python3-pip \
 && apt-get install -y python3-opencv \
 && python3 -m pip install --no-cache-dir --upgrade pip \
 # Installing dependencies
 && python3 -m pip install --no-cache-dir \
	opencv-python torchsampler rtpt torch argparse tqdm matplotlib torchvision \
      seaborn numpy \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH "${PYTHONPATH}:./"

WORKDIR "/root"
CMD ["/bin/bash"]