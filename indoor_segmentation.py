from pathlib import Path

import cv2
import numpy as np
import pixellib
import torch
from keras.preprocessing import image
from matplotlib import pyplot as plt
from pixellib.semantic import semantic_segmentation
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
IMG_DIR = ROOT_DIR / "images"
ROOT_AIRBNB_DIR = Path('/hdd2/past_students/virginia/airbnb/')
AIRBNB_IMG_DIR = ROOT_AIRBNB_DIR / "images"
#'/hdd2/past_students/virginia/hotel/images/expedia/1160' #oppure 88954
#/hdd2/past_students/virginia/hotel/images/traffickcam/10900
segment_image = semantic_segmentation()
segment_image.load_ade20k_model("/home/rozenberg/indoors_geolocation_pycharm/deeplabv3_xception65_ade20k.h5")
#segment_image.segmentAsAde20k("/content/drive/MyDrive/Hotels-50K/images/test/unoccluded/0/10050/traffickcam/3935054.jpg", output_image_name= "/content/drive/MyDrive/segmented.jpg")
img_orig = image.load_img(ROOT_DIR/"images/expedia/1160/2940274.jpg", target_size=(224, 224))
img_orig = ToTensor()(img_orig)
save_image(img_orig, '/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274_resized.jpg')
segmented_img = segment_image.segmentAsAde20k('/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274_resized.jpg', output_image_name= "/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274.jpg")
segmented_overlay = segment_image.segmentAsAde20k('/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274_resized.jpg', output_image_name= "/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274_overlay.jpg", overlay=True)
#plt.imshow("/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274_overlay.jpg")
print(segmented_img[1].shape)
print(segmented_img[0]['class_ids'])
print(segmented_img[0]["class_names"])
#print(segmented_overlay["class_ids"])
#print(segmented_overlay["class_names"])
torch.save(segmented_img, '/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274.pth')
num_segments = len(segmented_img[0]["class_ids"])
print(num_segments)
sgmnt_img = cv2.imread("/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274.jpg")
plt.imshow(sgmnt_img)
plt.show()
ovrl_img = cv2.imread("/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274_overlay.jpg")
plt.imshow(ovrl_img)
plt.show()

