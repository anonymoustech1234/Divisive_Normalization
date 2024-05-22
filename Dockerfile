# image based on [pytorch: 2.0, cuda: 11.7, cudnn: 8.0]
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# install system dependencies
RUN apt-get update \
 && apt-get install wget tar nano vim git tree g++ -y

# install python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    notebook \
    jupyterlab \
    tqdm \
    pyyaml \
    tensorboard \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn \
    pytorch-lightning \
    nvidia-dali-cuda110

CMD "/bin/bash"
