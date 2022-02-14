from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from models.airbnb_swinde20k import SwinTransformerFineTuningADE20k
from models.swintransformer import SwinTransformerFineTuning
from models.airbnb_swintransformer import SwinTransformerFineTuning
from pretrained_models.swin_transformer.models import swin_transformer
from datasets.hotel50k import Hotelimages
from datasets.airbnb import Airbnbimages
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

ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
IMG_DIR = ROOT_DIR / "images"
ROOT_AIRBNB_DIR = Path('/hdd2/past_students/virginia/airbnb/')
AIRBNB_IMG_DIR = ROOT_AIRBNB_DIR / "images"


data = Airbnbimages('test')
#dataset = Hotelimages('test')
dl = DataLoader(data, batch_size=16) #al posto di dl metti img=dataset[0], togli sample = next..., ind = 0 non serve più
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/2/model_val_epoch_loss=10.04.ckpt',device_pretrained='cpu')
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/0/model_val_epoch_loss=5.14.ckpt', device_pretrained='cpu')
model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swindade20k/1/model_val_epoch_loss=6.20.ckpt', device_pretrained='cpu').eval()
#target_layer = [model.subregion_predictor, model.country_predictor, model.city_predictor]
target_airbnb_layer = [model.subregion_predictor, model.country_predictor, model.location_predictor]

#sample = next(iter(dl)) #next fa iterazioni su iter iteratore
#sample = dataset[289]
index = data.dataset.index[data.dataset['image_id'] == 'rome_678'][0]
sample = data[index]

ind = 0

def attribute_image_features(algorithm, input, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=sample['y_country'],
                                              **kwargs
                                              )

    return tensor_attributions


#ig = IntegratedGradients(lambda x: model.forward_ig(x)['country_hat'])
#ig = GradientShap(lambda x: model.forward_ig(x)['subregion_hat'])
ig = Saliency(lambda x: model.forward_ig(x)['country_hat'])
input = sample['image'].unsqueeze(0)
input.requires_grad = True
#attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True) #n_steps=100
attr_ig = attribute_image_features(ig, input) #per saliency non usa baseline
attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
#print('Approximation delta: ', abs(delta)) decommenta con IntegratedGradient

print('Original Image')

original_image = np.transpose((sample['image'].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))

fig_orig, axes_orig = viz.visualize_image_attr(None, original_image,
                      method="original_image", title="Original Image")

fig_ig, axes_ig = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="all",
                         show_colorbar=True, title="Overlayed Integrated Gradients")
fig_orig.savefig('/home/rozenberg/indoors_geolocation_pycharm/integrated_gradients/original_airbnb_porto_152.png')
fig_ig.savefig('/home/rozenberg/indoors_geolocation_pycharm/integrated_gradients/ig_airbnb_subregion_saliency_porto_152.png')
