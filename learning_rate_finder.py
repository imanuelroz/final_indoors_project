import torch
from path import Path
from torch.utils.data import DataLoader
from models.swintransformer import SwinTransformerFineTuning
from models.swintransformerade20k import SwinTransformerFineTuningADE20k
from datasets.hotel50k import Hotelimages
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything
import numpy as np
import matplotlib.pyplot as plt


#root = Path(__file__).parent #__file__ mi da il percorso del file (mentre __name__ mi da il nome del file)
root = Path('/hdd2/indoors_geolocation_weights')
run_folder = root/'swinADE20k/auto_lr_rate' #swin_without_city
if not run_folder.exists():
    run_folder.mkdir()

def get_experiment_folder():
    i = 0
    while (run_folder/str(i)).exists():
        i += 1
    result = run_folder/str(i)
    result.mkdir()
    return result

def main():
    seed_everything(seed=13)
    exp_folder = get_experiment_folder()
    train_dataset = Hotelimages(split='train')
    target_list = []
    for t in train_dataset.dataset.country_id:
        target_list.append(t)
    target_list = torch.tensor(target_list).long()
    target_list = target_list[torch.randperm(len(target_list))]
    country_classes = 183
    c, _ = np.histogram(target_list.numpy(), bins=np.arange(0, country_classes + 1)) #put bins instead of _ if you want the plot
    #_ = plt.hist(target_list.numpy(), bins)
    #plt.show()
    weights = torch.tensor(np.true_divide(c, np.sum(c))) #usa sum invece di len
    class_weights_all = weights[target_list]
    #weights = train_dataset.dataset.subregion_id.apply(lambda x: 1/subregion_freq[x]).tolist()
    valid_dataset = Hotelimages(split='valid')
    batch_size = 16
    train_dataset_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_all, num_samples=len(train_dataset),
                                                                   replacement=False)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_dataset_sampler)
    #valid_dataset_sampler = torch.utils.data.RandomSampler(valid_dataset, replacement=False) #metti dentro anche num_samples=1600 e replacement=True
    #valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=valid_dataset_sampler)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    #model = SwinTransformerFineTuning()
    model = SwinTransformerFineTuningADE20k()
    model_checkpoint = ModelCheckpoint(exp_folder, monitor="val_epoch_loss", save_last=True, save_top_k=2,
                                       filename='model_{val_epoch_loss:.2f}', save_weights_only=False, every_n_epochs=1)
    model_es = EarlyStopping(monitor="val_epoch_loss")
    trainer = pl.Trainer(precision=16, default_root_dir=exp_folder, callbacks=[model_checkpoint, model_es],
                         max_epochs=10, accelerator='gpu', devices=1, auto_lr_find=False) #add model_es inside callbacks
    lr_finder = trainer.tuner.lr_find(model, train_dl, valid_dl, update_attr=True)
    print(lr_finder.results)
    fig = lr_finder.plot(suggest=True)
    fig.show()
    new_lr = lr_finder.suggestion()
    model.hparams.lr = new_lr
    print(f'Auto-find model LR: {model.hparams.lr}')
    trainer.fit(model, train_dl, valid_dl)



if __name__ == '__main__':    #__name__ è una variabile speciale che di solito contiene il nome del file
     main()                         # in questo caso contiene __main__ che è il punto di partenza
