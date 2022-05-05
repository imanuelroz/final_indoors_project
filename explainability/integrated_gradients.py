import torch
from matplotlib import pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
#from models.swintransformerade20k import SwinTransformerFineTuningADE20k
from models.airbnb_swinde20k import SwinTransformerFineTuningADE20k
from models.swintransformer import SwinTransformerFineTuning
from models.airbnb_swintransformer import SwinTransformerFineTuning
from pretrained_models.swin_transformer.models import swin_transformer
from datasets.hotel50k import Hotelimages
from datasets.airbnb import Airbnbimages, AirbnbimagesIG
import cv2
from path import Path
import numpy as np
import os
from captum.attr import IntegratedGradients, GradientShap
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from torch.utils.data import DataLoader
from keras.preprocessing import image

ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
IMG_DIR = ROOT_DIR / "images"
ROOT_AIRBNB_DIR = Path('/hdd2/past_students/virginia/airbnb/')
AIRBNB_IMG_DIR = ROOT_AIRBNB_DIR / "images"


data = Airbnbimages('train')
#data = Hotelimages('train')
dl = DataLoader(data, batch_size=16) #al posto di dl metti img=dataset[0], togli sample = next..., ind = 0 non serve più
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/2/model_val_epoch_loss=10.04.ckpt', device_pretrained='cpu').eval()
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/3/model_val_epoch_loss=9.16.ckpt', device_pretrained='cpu').eval()
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/indoors_geolocation_weights/swin_b/43/model_val_epoch_loss=11.60.ckpt', device_pretrained='cpu').eval()
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/0/model_val_epoch_loss=5.14.ckpt', device_pretrained='cpu').eval()
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swinade20k/unfreezed/3/model_val_epoch_loss=4.63.ckpt', device_pretrained='cpu').eval()
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swindade20k/1/model_val_epoch_loss=6.20.ckpt', device_pretrained='cpu').eval()
model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/19/model_val_epoch_loss=6.05.ckpt', device_pretrained = 'cpu').eval()
#target_layer = [model.subregion_predictor, model.country_predictor, model.city_predictor]
target_airbnb_layer = [model.subregion_predictor, model.country_predictor, model.location_predictor]

#sample = next(iter(dl)) #next fa iterazioni su iter iteratore
#sample = data.dataset['image_id']
#print(sample)
index = data.dataset.index[data.dataset['image_id'] == 'rome_400'][0]
#element = data.dataset.loc[data.dataset['image_id'] == 'bergamo_7']
#element = data.dataset.loc[data.dataset['image_id'] == 6549605]
#index = data.dataset.index[data.dataset['image_id'] == 6985480][0]
#index = data.dataset.loc['image_id' == '3834110']
print(index)
sample = data[index]
ind = 0

def attribute_image_features(algorithm, input, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=sample['y_location'],
                                              **kwargs
                                              )

    return tensor_attributions


ig = IntegratedGradients(lambda x: model.forward_ig(x)['location_hat'])
#ig = GradientShap(lambda x: model.forward_ig(x)['subregion_hat'])
#ig = Saliency(lambda x: model.forward_ig(x)['country_hat'])
input = sample['image'].unsqueeze(0) #con index
input.requires_grad = True
#attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True) #n_steps=100
attr_ig = attribute_image_features(ig, input) #per saliency non usa baseline
attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
#print('Approximation delta: ', abs(delta)) decommenta con IntegratedGradient

print('Original Image')

original_image = np.transpose((sample['image'].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
print(original_image.shape)
print(attr_ig.shape)
fig_orig, axes_orig = viz.visualize_image_attr(None, original_image,
                      method="original_image", title="Original Image")

fig_ig, axes_ig = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="absolute_value",
                         show_colorbar=True, title="Overlayed Integrated Gradients", cmap='plasma')
fig_orig.savefig('/home/rozenberg/indoors_geolocation_pycharm/integrated_gradients/rome_400.png')
fig_ig.savefig('/home/rozenberg/indoors_geolocation_pycharm/integrated_gradients/ig_rome_400_city.png')

#plt.close()
#plt.imshow(original_image)
#plt.imshow(np.abs(attr_ig), cmap='plasma', alpha=0.5)
#plt.show()