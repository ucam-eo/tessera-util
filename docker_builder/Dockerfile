FROM rocm/pytorch:latest
RUN python3 -m pip install wandb matplotlib imageio fiona rasterio scikit-learn opencv-python

# 设置工作目录
WORKDIR /workdir

# mkdir ~/docker_builder
# cp /media/frankfeng/btfm4rs/docker_builder/Dockerfile ~/docker_builder/
# cd ~/docker_builder
# docker build -t my-rocm .