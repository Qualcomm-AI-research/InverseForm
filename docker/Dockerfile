FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN pip install numpy
RUN pip install runx==0.0.6
RUN pip install sklearn
RUN pip install h5py
RUN pip install jupyter
RUN pip install scikit-image
RUN pip install pillow
RUN pip install piexif
RUN pip install cffi
RUN pip install tqdm
RUN pip install dominate
RUN pip install opencv-python
RUN pip install nose
RUN pip install ninja
RUN pip install fire

RUN apt-get update
RUN apt-get install libgtk2.0-dev -y && rm -rf /var/lib/apt/lists/*

# Install Apex
RUN cd /home/ && git clone https://github.com/NVIDIA/apex.git apex && cd apex && python setup.py install --cuda_ext --cpp_ext
WORKDIR /home/

RUN apt-get update \
    && apt-get install -y wget curl sudo software-properties-common

# Add sudo support
RUN echo "%users ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers
