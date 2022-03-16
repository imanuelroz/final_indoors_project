import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from path import Path
import gradio as gr
from pytorch_grad_cam.utils.image import preprocess_image
from torch.utils.data import DataLoader
from models.airbnb_swintransformer import SwinTransformerFineTuning
from models.vgg19 import Vgg_19_FineTuning
from datasets.hotel50k import Hotelimages
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import gradio as gr
import tensorflow as tf
import requests

root = Path('/hdd2/indoors_geolocation_weights')
run_folder = root / 'run'
ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
IMG_DIR = ROOT_DIR / "images"

model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/19/model_val_epoch_loss=6.05.ckpt', device_pretrained='cpu') # load the model
model.eval()
# Download human-readable labels for ImageNet.
country_ids = pd.read_csv('/hdd2/past_students/virginia/airbnb/airbnb_data/country_id.csv', index_col='Id')

@torch.no_grad()
def classify_image(inp):
  #inp = inp.reshape((-1, 3, 224, 224))
  print(inp.shape)
  preprocess = transforms.Compose([
    transforms.ToPILImage(), #togli se carichi con Image.Open
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  inp = preprocess(inp).unsqueeze(0)
  print(inp.shape)

  prediction = model(inp)['country_hat'].softmax(dim=-1)
  return {country_ids.loc[i, 'Country']: float(prediction[0, i]) for i in range(len(country_ids))}

image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=10)

gr.Interface(fn=classify_image, inputs=image, outputs=label, interpretation='shap').launch(share=True)
