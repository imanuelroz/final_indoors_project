import torch
from path import Path
from torch.utils.data import DataLoader
from models.airbnb_swintransformer import SwinTransformerFineTuning
from models.airbnb_swinde20k import SwinTransformerFineTuningADE20k
from datasets.airbnb import Airbnbimages
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

#root = Path(__file__).parent #__file__ mi da il percorso del file (mentre __name__ mi da il nome del file)
root = Path('/hdd2/airbnb_geolocation_weights')
run_folder = root/'swinade20k/unfreezed'
if not run_folder.exists():
    run_folder.mkdir()       #lo fai così perchè se crei con mkdir una cartella che già esiste ti da errore

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
    train_dataset = Airbnbimages(split='train') #prima c'era shuffle = True
    valid_dataset = Airbnbimages(split='valid') #ho aggiunto shuffle=True
    batch_size = 16 #metti 16 poi
    #train_dataset_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_all, num_samples=len(train_dataset),replacement=False)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #metti shuffle=False, sampler=train_dataset_sampler
    #valid_dataset_sampler = torch.utils.data.RandomSampler(valid_dataset, num_samples=1600, replacement=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) # shuffle=False, sampler=valid_dataset_sampler
    model = SwinTransformerFineTuningADE20k()
    model_checkpoint = ModelCheckpoint(exp_folder, monitor="val_epoch_loss", save_last=True, save_top_k=2,
                                       filename='model_{val_epoch_loss:.2f}', save_weights_only=False, every_n_epochs=2)
    model_es = EarlyStopping(monitor="val_epoch_loss")
    trainer = pl.Trainer(precision=16, default_root_dir=exp_folder, callbacks=[model_checkpoint, model_es],
                         max_epochs=80, accelerator='gpu', devices=1) #add model_es inside callbacks
    trainer.fit(model, train_dl, valid_dl)
    #new_model = model.load_from_checkpoint(checkpoint_path=run_folder/'1/model_epoch=14.ckpt')
    #trainer.fit(new_model, train_dl, valid_dl)

if __name__ == '__main__':    #__name__ è una variabile speciale che di solito contiene il nome del file
     main()                         # in questo caso contiene __main__ che è il punto di partenza