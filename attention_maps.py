from matplotlib import pyplot as plt
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
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from torch.utils.data import DataLoader

ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
IMG_DIR = ROOT_DIR / "images"
ROOT_AIRBNB_DIR = Path('/hdd2/past_students/virginia/airbnb/')
AIRBNB_IMG_DIR = ROOT_AIRBNB_DIR / "images"


data = Airbnbimages('train')
#dataset = Hotelimages('test')
dl = DataLoader(data, batch_size=16) #al posto di dl metti img=dataset[0], togli sample = next..., ind = 0 non serve più
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/2/model_val_epoch_loss=10.04.ckpt',device_pretrained='cpu')
model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/0/model_val_epoch_loss=5.14.ckpt', device_pretrained='cpu')
#target_layer = [model.subregion_predictor, model.country_predictor, model.city_predictor]
target_airbnb_layer = [model.subregion_predictor, model.country_predictor, model.location_predictor]

#sample = next(iter(dl)) #next fa iterazioni su iter iteratore
#sample = dataset[289]
index = data.dataset.index[data.dataset['image_id'] == 'rome_4963'][0]
sample = data[index]
model = model.eval()
model(sample['image'].unsqueeze(0))
attention_maps = []
for module in model.modules():
    #print(module)
    if hasattr(module,'attention_patches'):  #controlla se la variabile ha l' attributo
        print(module.attention_patches.shape)
        if module.attention_patches.numel() == 224*224:
            attention_maps.append(module.attention_patches)
for attention_map in attention_maps:
    attention_map = attention_map.reshape(224, 224, 1)
    plt.imshow(sample['image'].permute(1, 2, 0), interpolation='nearest')
    plt.imshow(attention_map, alpha=0.7, cmap=plt.cm.Greys)
    plt.show()

