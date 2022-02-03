FROM pytorch/conda-cuda

RUN pip3 install pandas numpy pillow sklearn timm path pytorch-lightning torchvision
COPY . /
ENV CUDA_VISIBLE_DEVICE=0

CMD ["python3", "train.py"]