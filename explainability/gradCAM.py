import torch
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from models.swintransformer import SwinTransformerFineTuning
from models.airbnb_swintransformer import SwinTransformerFineTuning
from models.airbnb_swinde20k import SwinTransformerFineTuningADE20k
from pretrained_models.swin_transformer.models import swin_transformer
from datasets.hotel50k import Hotelimages
import cv2
from path import Path
import numpy as np
import os

ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
IMG_DIR = ROOT_DIR / "images"
ROOT_AIRBNB_DIR = Path('/hdd2/past_students/virginia/airbnb/')
AIRBNB_IMG_DIR = ROOT_AIRBNB_DIR / "images"

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(1))
    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/3/model_val_epoch_loss=9.17.ckpt').eval()
model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swindade20k/1/model_val_epoch_loss=6.20.ckpt').eval()# device_pretrained='cpu')
original_forward = model.forward_ig

def forward_wrapper(x):
    y = original_forward(x)
    return y['country_hat']
model.forward=forward_wrapper

target_layer = [model.subregion_predictor]# model.country_predictor, model.city_predictor]
cam = FullGrad(model=model, target_layers=target_layer)# reshape_transform=reshape_transform)
#rgb_img = cv2.imread(IMG_DIR/"expedia/51098/6969621.jpg")[:, :, ::-1]
#rgb_img = cv2.imread(AIRBNB_IMG_DIR/"porto/porto_152.jpg")[:, :, ::-1] #decommenta, ho commentato solo per fare inferenza su immagine segmentata
rgb_img = cv2.imread("/hdd2/airbnb_geolocation_weights/segmented.jpeg")
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
plt.imshow(rgb_img)
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

input_tensor.requires_grad_(True)
grayscale_cam = cam(input_tensor=input_tensor,
                    eigen_smooth=True,
                    aug_smooth=True)
print(grayscale_cam.shape, grayscale_cam.dtype)
grayscale_cam = grayscale_cam[0, :]
#grayscale_cam = torch.from_numpy(grayscale_cam)
#grayscale_cam = grayscale_cam.flatten().softmax(0).reshape(1, 224, 224).numpy().astype('float32')
print(grayscale_cam)
cam_image = show_cam_on_image(rgb_img, grayscale_cam)
directory = Path('/home/rozenberg/indoors_geolocation_pycharm/gradCAM_images')
os.chdir(directory)
cv2.imwrite('save_img.jpg', cam_image)
plt.imshow(grayscale_cam, alpha=0.5, cmap='plasma') #senza cmap è più tradizionale come visual
plt.show()

