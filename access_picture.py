from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from models.swintransformer import SwinTransformerFineTuning
from models.airbnb_swintransformer import SwinTransformerFineTuning
from pretrained_models.swin_transformer.models import swin_transformer
from datasets.hotel50k import Hotelimages
from datasets.airbnb import Airbnbimages
import cv2
from path import Path
import numpy as np
import os
# Boilerplate imports.
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import torch
from torchvision import models, transforms

# From our repository.
import saliency.core as saliency



ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
IMG_DIR = ROOT_DIR / "images"
ROOT_AIRBNB_DIR = Path('/hdd2/past_students/virginia/airbnb/')
AIRBNB_IMG_DIR = ROOT_AIRBNB_DIR / "images"


data = Airbnbimages('test')

#dataset = Hotelimages('test')
#dl = DataLoader(dataset, batch_size=16) #al posto di dl metti img=dataset[0], togli sample = next..., ind = 0 non serve più
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/2/model_val_epoch_loss=10.04.ckpt',device_pretrained='cpu')
model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/19/model_val_epoch_loss=6.05.ckpt', device_pretrained='cpu')
#target_layer = [model.subregion_predictor, model.country_predictor, model.city_predictor]
target_airbnb_layer = [model.subregion_predictor, model.country_predictor, model.location_predictor]

#sample = next(iter(dl)) #next fa iterazioni su iter iteratore
index = data.dataset.index[data.dataset['image_id']=='bergamo_56'][0]
print(index)
#sample = dataset[index]
#print(index)