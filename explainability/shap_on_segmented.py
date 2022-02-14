import torch
#from matplotlib import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from sklearn.feature_selection.tests.test_base import feature_names
from torchvision import transforms
from pretrained_models.swin_transformer.models import swin_transformer
from models.swintransformer import SwinTransformerFineTuning
#from models.airbnb_swintransformer import SwinTransformerFineTuning
from models.airbnb_swinde20k import SwinTransformerFineTuningADE20k
from models.swintransformerade20k import SwinTransformerFineTuningADE20k
from datasets.hotel50k import Hotelimages
import cv2
from path import Path
import numpy as np
from keras.preprocessing import image
from skimage.segmentation import slic
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import shap
import warnings
# load model data

root = Path('/hdd2/indoors_geolocation_weights')
run_folder = root / 'run'
ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
IMG_DIR = ROOT_DIR / "images"

#model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swindade20k/1/model_val_epoch_loss=6.20.ckpt').eval()
model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/3/model_val_epoch_loss=9.17.ckpt').eval()
#model.eval()
# Download human-readable labels for ImageNet.
#country_ids = pd.read_csv('/hdd2/past_students/virginia/airbnb/airbnb_data/country_id.csv', index_col='Id')
country_ids = pd.read_csv('/hdd2/past_students/virginia/hotel/data/country_ids.csv', index_col='country_id')
original_img = cv2.imread(ROOT_DIR/"images/expedia/1160/2940274.jpg")
#rgb_img = cv2.resize(rgb_img, (224, 224))
#rgb_img = np.float32(rgb_img) / 255
#plt.imshow(rgb_img)
#input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                         #std=[0.229, 0.224, 0.225])
plt.axis("off")


segm_img = cv2.imread("/home/rozenberg/indoors_geolocation_pycharm/segmented_indoor_scenes/2940274.jpg")
#segm_img = cv2.resize(segm_img, (224, 224))
#rgb_img = np.float32(rgb_img) / 255
#plt.imshow(segm_img)
#plt.show()

def preprocess_image(inp):
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

rgb_img = preprocess_image(original_img)
segm_img = preprocess_image(segm_img)

#Define a function that depends ona  binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background
    return out


def f(z):
    prediction = model((mask_image(z, segm_img, rgb_img, 255)))['country_hat'].softmax(dim=-1)
    #return {country_ids.loc[i, 'Country']: float(prediction[0, i]) for i in range(len(country_ids))}

# use Kernel SHAP to explain the network's predictions
explainer = shap.KernelExplainer(f, np.zeros((1, 12)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    shap_values = explainer.shap_values(np.ones((1, 12)), nsamples=1000)

# get the top predictions from the model
preds = model.predict((np.expand_dims(rgb_img.copy(), axis=0)))
top_preds = np.argsort(-preds)
inds = top_preds[0]
top_10_pred = pd.Series(data={feature_names[str(inds[i])][1]:preds[0, inds[i]] for i in range(10)})
top_10_pred.plot(kind='bar', title='Top 10 Predictions')





