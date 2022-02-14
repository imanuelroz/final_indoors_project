import argparse

import torch
import yaml
from path import Path
from torch.utils.data import DataLoader
#from models.airbnb_swintransformer import SwinTransformerFineTuning
from models.airbnb_swinde20k import SwinTransformerFineTuningADE20k
from models.vgg19 import Vgg_19_FineTuning
from datasets.airbnb import Airbnbimages
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# root = Path(__file__).parent #__file__ mi da il percorso del file (mentre __name__ mi da il nome del file)
root = Path('/hdd2/indoors_geolocation_weights')
run_folder = root / 'run'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=Path, dest='run_path', required=True)
    args = parser.parse_args()
    test_dataset = Airbnbimages(split='test')
    # test_dataset_sampler = torch.utils.data.RandomSampler(test_dataset, num_samples=1600, replacement=True) #change num_samples
    batch_size = 16  # metti 16 poi
    # test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_dataset_sampler) #poi rimetti shuffle True
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #model = SwinTransformerFineTuning.load_from_checkpoint(args.run_path)
    model = SwinTransformerFineTuningADE20k.load_from_checkpoint(args.run_path)
    #model = Vgg_19_FineTuning.load_from_checkpoint(args.run_path)  # statt accuort ha da fa aaccussi senno vedi linea 34
    trainer = pl.Trainer(precision=16, default_root_dir=args.run_path.parent,
                         max_epochs=10, accelerator='gpu', devices=1, max_steps=100)
    # model.load_from_checkpoint(args.run_path)
    # model.load_state_dict(torch.load(args.run_path)['state_dict'])
    trainer.test(model, test_dl)

    results = dict(location_top_1_accuracy=model.test_avg_top_1_accuracy_location.compute().item(),
                   subregion_top_1_accuracy=model.test_avg_top_1_accuracy_subregion.compute().item(),
                   country_top_1_accuracy=model.test_avg_top_1_accuracy_country.compute().item(),
                   location_top_5_accuracy=model.test_avg_top_5_accuracy_location.compute().item(),
                   subregion_top_5_accuracy=model.test_avg_top_5_accuracy_subregion.compute().item(),
                   country_top_5_accuracy=model.test_avg_top_5_accuracy_country.compute().item(),
                   location_top_10_accuracy=model.test_avg_top_10_accuracy_location.compute().item(),
                   subregion_top_10_accuracy=model.test_avg_top_10_accuracy_subregion.compute().item(),
                   country_top_10_accuracy=model.test_avg_top_10_accuracy_country.compute().item()
                   )
    print(results)
    with open(args.run_path + '_accuracy.yaml', 'w') as f:
        yaml.safe_dump(results, f)


if __name__ == '__main__':
    main()

# se fai il test con metriche computate su vecchi modelli togli dropout da forward
