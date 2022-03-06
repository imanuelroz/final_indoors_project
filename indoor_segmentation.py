import os.path
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

def modelSegmentation(typeDir,folderId,imgId):

    segment_image = semantic_segmentation()
    segment_image.load_ade20k_model("/home/rozenberg/indoors_geolocation_pycharm/deeplabv3_xception65_ade20k.h5")

    if(typeDir == 0): #typeDir == 0 means AirBnb
        ROOT_DIR = Path('/hdd2/past_students/virginia/airbnb/')
        IMG_DIR = ROOT_DIR / "images"
        img_orig = image.load_img(IMG_DIR / str(str(folderId) + "/" + str(folderId) + "_" + str(imgId) + ".jpg") , target_size=(224, 224))
    else: #typeDir == 1,2 means Expedia/Traffickcam
        ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
        if(typeDir == 1): #typeDir == 1 means Expedia
            IMG_DIR = ROOT_DIR / "images/expedia"
        else:#typeDir == 2 means Traffickcam
            IMG_DIR = ROOT_DIR / "images/traffickcam"
        img_orig = image.load_img(IMG_DIR / str(str(folderId) + "/" + str(imgId) + ".jpg") , target_size=(224, 224))

    img_orig = ToTensor()(img_orig)

    save_image(img_orig, '/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/resized_'+str(imgId)+'.jpg')
    segmented_img = segment_image.segmentAsAde20k(
        '/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/resized_'+str(imgId)+'.jpg',
        output_image_name="/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/segmented_"+str(imgId)+".jpg")
    segmented_overlay = segment_image.segmentAsAde20k(
        '/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/resized_'+str(imgId)+'.jpg',
        output_image_name="/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/overlay_"+str(imgId)+".jpg",
        overlay=True)

    print(segmented_img[1].shape)
    print(segmented_img[0]['class_ids'])
    print(segmented_img[0]["class_names"])

    pthPath = '/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/'+str(folderId) + "_" + str(imgId)+'.pth'

    torch.save(segmented_img, pthPath)
    num_segments = len(segmented_img[0]["class_ids"])
    print(num_segments)
    sgmnt_img = cv2.imread("/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/segmented_"+str(imgId)+".jpg")
    plt.imshow(sgmnt_img)
    plt.show()
    ovrl_img = cv2.imread("/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/overlay_"+str(imgId)+".jpg")
    plt.imshow(ovrl_img)
    plt.show()

    return torch.load(pthPath)

'''
typeDir = 2
folderId = 886
imgId = 3898717

airbnb_img = modelSegmentation(typeDir, folderId, imgId)
'''

