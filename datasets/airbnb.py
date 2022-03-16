import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from path import Path
import torchvision
from PIL import Image, ImageEnhance, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT_DIR = Path('/hdd2/past_students/virginia/airbnb/')
AIRBNB_DATA_DIR = ROOT_DIR / "airbnb_data"
#AIRBNB_DIR = Path('/hdd2/past_students/virginia/airbnb')
#AIRBNB_DATA = AIRBNB_DIR / "data"
LOCAL_DATA_DIR = Path(__file__).parent.parent/'data'

class Airbnbimages(Dataset):
    def __init__(self, split: str):
        assert split in ['train', 'valid', 'test'], split  # così se è un errore mi stampa il valore di split sbagliato
        self.split_ids = pd.read_csv(AIRBNB_DATA_DIR / f'{split}.csv', low_memory=False)  # f'{}' fa l' interpolazione di stringhe
        self.airbnb_info = pd.read_csv(AIRBNB_DATA_DIR / 'filtered_data.csv', low_memory=False, usecols=['image_id'])
        self.images_path = pd.read_csv(AIRBNB_DATA_DIR / 'image_paths.csv', low_memory=False)
        #self.images_path = pd.read_csv(LOCAL_DATA_DIR / 'images_path_rgb_only.csv')
        self.dataset = pd.merge(self.airbnb_info, self.split_ids, on=['image_id'])
        self.dataset = pd.merge(self.dataset, self.images_path, how='left', left_on='image_id', right_on='Id')
        for c in ['country', 'location', 'subregion']:
            self.dataset[c] = pd.to_numeric(self.dataset[c])
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.Pad(144),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(self.dataset.columns.tolist())
        print(self.dataset.head())

    def __getitem__(self, index) -> T_co:   #indicizzazione
        image_path = self.dataset.loc[index, 'Path']
        y_country = int(self.dataset.loc[index, 'country'])
        y_location = int(self.dataset.loc[index, 'location'])
        y_subregion = int(self.dataset.loc[index, 'subregion']) #.values me lo converte in numpy
        #y = torch.LongTensor([y_country, y_location, y_subregion])
        image = Image.open(ROOT_DIR/'images'/image_path).convert('RGB') #momentaneamente così poi togli e filtra tutto
        image = self.preprocess(image)
        return dict(image=image, y_country=y_country, y_location=y_location, y_subregion=y_subregion)

    def __len__(self):
        return len(self.dataset)

class AirbnbimagesIG(Dataset):
    def __init__(self, split: str):
        assert split in ['train', 'valid', 'test'], split  # così se è un errore mi stampa il valore di split sbagliato
        self.split_ids = pd.read_csv(AIRBNB_DATA_DIR / f'{split}.csv', low_memory=False)  # f'{}' fa l' interpolazione di stringhe
        self.airbnb_info = pd.read_csv(AIRBNB_DATA_DIR / 'filtered_data.csv', low_memory=False, usecols=['image_id'])
        self.images_path = pd.read_csv(AIRBNB_DATA_DIR / 'image_paths.csv', low_memory=False)
        #self.images_path = pd.read_csv(LOCAL_DATA_DIR / 'images_path_rgb_only.csv')
        self.dataset = pd.merge(self.airbnb_info, self.split_ids, on=['image_id'])
        self.dataset = pd.merge(self.dataset, self.images_path, how='left', left_on='image_id', right_on='Id')
        for c in ['country', 'location', 'subregion']:
            self.dataset[c] = pd.to_numeric(self.dataset[c])
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.CenterCrop(224),
            #transforms.Pad(144),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(self.dataset.columns.tolist())
        print(self.dataset.head())

    def __getitem__(self, index) -> T_co:   #indicizzazione
        image_path = self.dataset.loc[index, 'Path']
        y_country = int(self.dataset.loc[index, 'country'])
        y_location = int(self.dataset.loc[index, 'location'])
        y_subregion = int(self.dataset.loc[index, 'subregion']) #.values me lo converte in numpy
        #y = torch.LongTensor([y_country, y_location, y_subregion])
        image = Image.open(ROOT_DIR/'images'/image_path).convert('RGB') #momentaneamente così poi togli e filtra tutto
        image = self.preprocess(image)
        print(image.shape)
        return dict(image=image, y_country=y_country, y_location=y_location, y_subregion=y_subregion)

    def __len__(self):
        return len(self.dataset)