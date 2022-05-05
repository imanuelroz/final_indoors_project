import torch
#from matplotlib import transforms
from matplotlib import cm
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from sklearn.feature_selection.tests.test_base import feature_names
from torchvision import transforms
from pretrained_models.swin_transformer.models import swin_transformer
from models.swintransformer import SwinTransformerFineTuning
from models.airbnb_swintransformer import SwinTransformerFineTuning
#from models.airbnb_swintransformer import SwinTransformerFineTuning
from models.airbnb_swinde20k import SwinTransformerFineTuningADE20k
from datasets.hotel50k import Hotelimages
import cv2
from path import Path
import numpy as np

from keras.preprocessing import image
import requests
from skimage.segmentation import slic
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import shap
import warnings
import indoor_segmentation as inS
# load model data

root = Path('/hdd2/indoors_geolocation_weights')
#run_folder = root / 'run'
#ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
#IMG_DIR = ROOT_DIR / "images"



ROOT_DIR = Path('/hdd2/past_students/virginia/airbnb/')
IMG_DIR = ROOT_DIR / "images"
AIRBNB_DATA_DIR = ROOT_DIR / "airbnb_data"
LOCAL_DATA_DIR = Path(__file__).parent.parent/'data'
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/0/model_val_epoch_loss=5.14.ckpt').eval()
model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/19/model_val_epoch_loss=6.05.ckpt', device_pretrained = 'cpu').eval()
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swinade20k/unfreezed/3/model_val_epoch_loss=4.63.ckpt').eval()
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/3/model_val_epoch_loss=9.17.ckpt').eval()
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/indoors_geolocation_weights/swinADE20k/4/model_val_epoch_loss=9.53.ckpt').eval()
# Download human-readable labels for ImageNet.


#subregion_ids = pd.read_csv('/hdd2/past_students/virginia/airbnb/airbnb_data/subregion_id.csv', index_col='subregion_id')
#country_ids = pd.read_csv('/hdd2/past_students/virginia/airbnb/airbnb_data/country_id.csv', index_col='Id')
location_ids = pd.read_csv('/hdd2/past_students/virginia/airbnb/airbnb_data/location_id.csv', index_col='Id')
img = image.load_img(ROOT_DIR/"images/rio/rio_720.jpg", target_size=(224, 224))
img_orig = image.img_to_array(img)

#if we want to use slic instead of segmented image

#segments_slic = slic(img, n_segments=100, compactness=30, sigma=3)
#plt.imshow(segments_slic)
#plt.show()
#plt.axis('off')


#img = image.load_img(ROOT_DIR/"images/expedia/1160/2940274.jpg", target_size=(224, 224))
#img_orig = image.img_to_array(img)
#sgm = image.load_img("/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274.jpg", target_size=(224, 224))
#sgm_img = image.img_to_array(sgm)

#sgm_dict = torch.load('/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274.pth')
sgm_dict = inS.modelSegmentation(0, "rio", 720)
sgm_img_rgb = sgm_dict[1]
#print(type(sgm_img_rgb))


class_colors = sgm_dict[0]['class_colors']
print(class_colors)

def getSector(pixel, class_colors):
    for i in range(len(class_colors)):
        tmp = np.array_equal(pixel, class_colors[i])
        #print(tmp)
        if tmp:
            #print("the index is: ", i)
            return i
    return -1

def parse_segmetnation(img):
    class_colors = img[0]['class_colors']

    for i in range(len(class_colors)):
        tmp = class_colors[i][2]
        class_colors[i][2] = class_colors[i][0]
        class_colors[i][0] = tmp
    #print(class_colors)

    img_seg = img[1]
    row = len(img_seg)
    col = len(img_seg[0])


    segmentation = np.empty((224,224),int)
    for i in range(row):
        for j in range(col):
            segmentation[i][j] = getSector(img_seg[i][j], class_colors)
    return segmentation


slic_style_mask = parse_segmetnation(sgm_dict)
#print(parse_segmetnation(sgm_dict))


sgm_img = torch.zeros(sgm_img_rgb.shape[0], sgm_img_rgb.shape[1])
#plt.imshow(sgm_img_rgb)
#plt.show()
print(sgm_dict[0]['masks'].shape)
for i in range(sgm_img_rgb.shape[0]):
    for j in range(sgm_img_rgb.shape[1]):
        print('----------------')
        for i_class, y_class in enumerate(sgm_dict[0]['class_ids']):
            print(sgm_img_rgb[i, j], sgm_dict[0]['class_colors'][i_class])
            if (sgm_img_rgb[i, j] == sgm_dict[0]['class_colors'][i_class]).all():
                print('Match', i_class)
                sgm_img[i, j] = i_class #y_class
                break
#print(sgm_img.max(), sgm_img.mean(), sgm_img.std())
#print(img_orig.shape)
#print(sgm_img_rgb.shape, 'shape di img_rgb')
plt.imshow(sgm_img_rgb)
plt.show()
plt.imshow(sgm_img)
plt.show()
#num_classes = max(sgm_dict[0]['class_ids'])+1 se usi y_class
num_classes = len(sgm_dict[0]['class_ids'])
#print("image_dimension: ", img_orig)
#Define a function that depends ona  binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))
    print(image.shape)
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]), dtype='float32')
    print(out.shape)
    #print(zs.shape[0])
    #print(zs.shape[1])
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                #print(segmentation == j)
                out[i][segmentation == j] = background #c' era una virgola dopo j che ho tolto non combaciava di dimensione
    masked_image_pytorch = torch.from_numpy(out)
    masked_image_pytorch = masked_image_pytorch.permute(0, 3, 1, 2)
    return masked_image_pytorch

@torch.no_grad() #per non tenere in memoria tutti ir isultati intermedi dei layer
def f(z):
    prediction = model(mask_image(z, slic_style_mask, img_orig, 255))['location_hat'].softmax(dim=-1) #sgm_img anziche slic_style_mask
    #print(prediction)
    return prediction.numpy()
    #return {location_ids.loc[i, 'location']: float(prediction[0, i]) for i in range(len(location_ids))}

def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out



masked_images = mask_image(torch.zeros((1, num_classes)), slic_style_mask, img_orig, 255)
print(masked_images.shape)

plt.imshow(masked_images[0][0, :, :])
plt.axis('off')
plt.show()

prediction = model((mask_image(torch.zeros(1, num_classes), slic_style_mask, img_orig, 255)))['location_hat'].softmax(dim=-1)
print(prediction)


# use Kernel SHAP to explain the network's predictions
explainer = shap.KernelExplainer(f, np.zeros((1, num_classes)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(np.ones((1, num_classes)), nsamples=100)


# get the top predictions from the model
img_orig_pytorch = torch.from_numpy(img_orig).unsqueeze(0)
img_orig_pytorch = img_orig_pytorch.permute(0, 3, 1, 2)
with torch.no_grad():
    preds = model(img_orig_pytorch)['location_hat'].softmax(dim=-1)
top_preds = np.argsort(-preds)
inds = top_preds[0]
#top_10_pred = pd.Series(data={feature_names[str(inds[i])][1]:preds[0, inds[i]] for i in range(10)})
#top_10_pred.plot(kind='bar', title='Top 10 Predictions')


# Visualize the explanations
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
inds = top_preds[0]
print(inds)
axes[0].imshow(img)
axes[0].axis('off')

max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])
for i in range(3):
    m = fill_segmentation(shap_values[inds[i]][0], sgm_img)
    print(location_ids)
    print(location_ids.loc[inds[i].item(), 'Location'])
    axes[i+1].set_title(location_ids.loc[inds[i].item(), 'Location'])
    axes[i+1].imshow(np.array(img.convert('LA'))[:, :, 0], alpha=0.9) #era 0.15
    im = axes[i+1].imshow(m, cmap='viridis', vmin=-max_val, vmax=max_val, alpha=0.8)
    axes[i+1].axis('off')
cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)
#plt.savefig('/home/rozenberg/indoors_geolocation_pycharm/rio_720_shap.jpg')
plt.show()
