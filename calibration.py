import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix, f1_score
from reliability_diagrms import compute_calibration, reliability_diagram
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
#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/run/19/model_val_epoch_loss=6.05.ckpt').eval().cuda() #.cuda() per mandare in gpu
model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swinade20k/unfreezed/best_lr_rate/2/model_val_epoch_loss=4.21.ckpt').eval().cuda()#model = SwinTransformerFineTuning.load_from_checkpoint('/hdd2/indoors_geolocation_weights/run/3/model_val_epoch_loss=9.17.ckpt').eval()
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint('/hdd2/airbnb_geolocation_weights/swinade20k/unfreezed/3/model_val_epoch_loss=4.63.ckpt').eval()
#model = Swin_b_TransformerFineTuning.load_from_checkpoint(args.run_path)
#model = Vgg_19_FineTuning.load_from_checkpoint(args.run_path)  # statt accuort ha da fa aaccussi senno vedi linea 34
#model = EfficientNet_FineTuning.load_from_checkpoint(args.run_path)
#model = SwinTransformerFineTuningADE20k.load_from_checkpoint(args.run_path)
trainer = pl.Trainer(precision=16,
                     max_epochs=10, accelerator='gpu', devices=1, max_steps=100)
# model.load_from_checkpoint(args.run_path)
# model.load_state_dict(torch.load(args.run_path)['state_dict'])
#trainer.test(model, test_dl)
y_country_probabilities = []
y_subregion_probabilities = []
y_location_probabilities = []
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
    y_subregion_probabilities.append(pred['subregion_hat'].cpu())
    y_country_probabilities.append(pred['country_hat'].cpu())
    y_location_probabilities.append(pred['location_hat'].cpu())

y_subregion_probabilities = torch.cat(y_subregion_probabilities, dim=0).cpu().softmax(dim=-1).numpy()
y_country_probabilities = torch.cat(y_country_probabilities, dim=0).cpu().softmax(dim=-1).numpy()
y_location_probabilities = torch.cat(y_location_probabilities, dim=0).cpu().softmax(dim=-1).numpy()
y_subregion_pred = torch.cat(y_subregion_pred, dim=0).cpu().numpy()
y_country_pred = torch.cat(y_country_pred, dim=0).cpu().numpy()
y_location_pred = torch.cat(y_location_pred, dim=0).cpu().numpy()
y_subregion = torch.cat(y_subregion, dim=0).cpu().numpy()
y_country = torch.cat(y_country, dim=0).cpu().numpy()
y_location = torch.cat(y_location, dim=0).cpu().numpy()
#f1_score_subregion = f1_score(y_subregion, y_subregion_pred, average=None)
#f1_score_country = f1_score(y_country, y_country_pred, average=None)
#f1_score_location = f1_score(y_location, y_location_pred, average=None)
#class_subregion = np.argmax(f1_score_subregion)
#class_country = np.argmax(f1_score_country)
#class_location = np.argmax(f1_score_location)

#compute_calibration(y_subregion==class_subregion, y_subregion_pred==class_subregion, confidences=y_subregion_probabilities[:,class_subregion])
subregion_diagram = reliability_diagram(y_subregion, y_subregion_pred, confidences=y_subregion_probabilities.max(axis=-1), return_fig=True)
subregion_diagram.savefig('/home/rozenberg/indoors_geolocation_pycharm/gradCAM_images/subregion_calibration.jpeg')
plt.show()
plt.close()
country_diagram = reliability_diagram(y_country, y_country_pred, confidences=y_country_probabilities.max(axis=-1), return_fig=True)
country_diagram.savefig('/home/rozenberg/indoors_geolocation_pycharm/gradCAM_images/country_calibration.jpeg')
plt.close()
location_diagram = reliability_diagram(y_location, y_location_pred, confidences=y_location_probabilities.max(axis=-1), return_fig=True)
location_diagram.savefig('/home/rozenberg/indoors_geolocation_pycharm/gradCAM_images/location_calibration.jpeg')
plt.close()