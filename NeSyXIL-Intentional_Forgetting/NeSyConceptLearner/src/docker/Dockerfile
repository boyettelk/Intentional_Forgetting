FROM nvcr.io/nvidia/pytorch:23.02-py3
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "zsh"]
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true