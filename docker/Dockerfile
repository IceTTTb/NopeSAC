FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&  apt-get install -y \
    git \
    build-essential \
    vim \
    openssh-server \
    python3-opencv \
    ca-certificates \
    python3-dev \
    python3-pip \
    wget \
    ninja-build \
    mesa-common-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    xauth

RUN sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config

RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN pip3 install numpy==1.23.4 
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install scipy numpy-quaternion scikit-learn seaborn 
# Install detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git && python -m pip install -e detectron2


RUN git clone https://github.com/IceTTTb/NopeSAC.git /root/NopeSAC

RUN  echo 'root:root' | chpasswd

RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

RUN service ssh start

CMD ["/usr/sbin/sshd","-D"]
