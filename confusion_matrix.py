import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix
#from matplotlib import transforms
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision import transforms
from pretrained_models.swin_transformer.models import swin_transformer
from models.swintransformer import SwinTransformerFineTuning
from models.airbnb_swintransformer import SwinTransformerFineTuning
#from models.airbnb_swintransformer import SwinTransformerFineTuning
from models.airbnb_swinde20k import SwinTransformerFineTuningADE20k
#from datasets.hotel50k import Hotelimages
from datasets.airbnb import Airbnbimages
import cv2
from path import Path
import numpy as np
import matplotlib.pylab as plt
import numpy as np
import pandas as pd



test_dataset = Airbnbimages(split='test')
#test_dataset_sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=160, replacement=True) #change num_samples
batch_size = 16  # metti 16 poi
#test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_dataset_sampler) #poi rimetti shuffle True
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/19/model_val_epoch_loss=6.05.ckpt').eval().cuda()
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/3/model_val_epoch_loss=9.17.ckpt').eval()
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swinade20k/unfreezed/3/model_val_epoch_loss=4.63.ckpt').eval()
#model = Swin_b_TransformerFineTuning.load_from_checkpoint(args.run_path)
#model = Vgg_19_FineTuning.load_from_checkpoint(args.run_path)
#model = EfficientNet_FineTuning.load_from_checkpoint(args.run_path)
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint(args.run_path)
trainer = pl.Trainer(precision=16,
                     max_epochs=10, accelerator='gpu', devices=1, max_steps=100)
# model.load_from_checkpoint(args.run_path)
# model.load_state_dict(torch.load(args.run_path)['state_dict'])
#trainer.test(model, test_dl)

y_subregion_pred = []
y_country_pred = []
y_location_pred = []
y_subregion = []
y_country = []
y_location = []
for batch in test_dl:
    for k, v in batch.items():
        batch[k] = v.cuda()
    with torch.no_grad():
        pred = model(batch['image'])
    y_subregion_pred.append(pred['subregion_hat'].argmax(dim=-1).cpu())
    y_country_pred.append(pred['country_hat'].argmax(dim=-1).cpu())
    y_location_pred.append(pred['location_hat'].argmax(dim=-1).cpu())
    y_subregion.append(batch['y_subregion'].cpu())
    y_country.append(batch['y_country'].cpu())
    y_location.append(batch['y_location'].cpu())

y_subregion_pred = torch.cat(y_subregion_pred, dim=0).cpu()
y_country_pred = torch.cat(y_country_pred, dim=0).cpu()
y_location_pred = torch.cat(y_location_pred, dim=0).cpu()
y_subregion = torch.cat(y_subregion, dim=0).cpu()
y_country = torch.cat(y_country, dim=0).cpu()
y_location = torch.cat(y_location, dim=0).cpu()

confusion_subregion = confusion_matrix(y_subregion, y_subregion_pred)
confusion_country = confusion_matrix(y_country, y_country_pred)
confusion_location = confusion_matrix(y_location, y_location_pred)

plt.imshow(confusion_country)
plt.title('Confusion Matrix of Country')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.savefig('/home/rozenberg/indoors_geolocation_pycharm/gradCAM_images/confusion_country.jpeg')
plt.show()
plt.close()

plt.imshow(confusion_subregion)
plt.title('Confusion Matrix of Subregion')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.savefig('/home/rozenberg/indoors_geolocation_pycharm/gradCAM_images/confusion_subregion.jpeg')
plt.show()
plt.close()

plt.imshow(confusion_location)
plt.title('Confusion Matrix of Location')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.savefig('/home/rozenberg/indoors_geolocation_pycharm/gradCAM_images/confusion_location.jpeg')
plt.show()
plt.close()








