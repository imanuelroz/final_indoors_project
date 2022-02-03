import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from path import Path
import torchvision
from PIL import Image, ImageEnhance, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
DATA_DIR = ROOT_DIR / "data"
#AIRBNB_DIR = Path('/hdd2/past_students/virginia/airbnb')
#AIRBNB_DATA = AIRBNB_DIR / "data"
LOCAL_DATA_DIR = Path(__file__).parent.parent/'data'

class Hotelimages(Dataset):
    def __init__(self, split: str):
        assert split in ['train', 'valid', 'test'], split  # così se è un errore mi stampa il valore di split sbagliato
        self.split_ids = pd.read_csv(DATA_DIR / f'{split}.csv')  # f'{}' fa l' interpolazione di stringhe
        self.hotel_info = pd.read_csv(DATA_DIR / 'hotel_info.csv')
        self.images_path = pd.read_csv(DATA_DIR / 'images_path.csv')    #toglilo rimesso solo per converto to rgb
        #self.images_path = pd.read_csv(LOCAL_DATA_DIR / 'images_path_rgb_only.csv')
        self.dataset = pd.merge(self.split_ids, self.hotel_info, on='hotel_id')
        self.dataset = pd.merge(self.dataset, self.images_path, on='image_id')
        for c in ['chain_id', 'country_id', 'city_id', 'subregion_id']:
            self.dataset[c] = pd.to_numeric(self.dataset[c])
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   #poi togli il commento ricordati
        ])
        print(self.dataset.columns.tolist())
        print(self.dataset.head())

    def __getitem__(self, index) -> T_co:   #indicizzazione
        image_path = self.dataset.loc[index, 'path']
        y_chain = int(self.dataset.loc[index, 'chain_id'])
        y_country = int(self.dataset.loc[index, 'country_id'])
        y_city = int(self.dataset.loc[index, 'city_id'])
        y_subregion = int(self.dataset.loc[index, 'subregion_id']) #.values me lo converte in numpy
        #y = torch.LongTensor([y_country, y_city, y_subregion])
        image = Image.open(ROOT_DIR/image_path).convert('RGB') #momentaneamente così poi togli e filtra tutto
        image = self.preprocess(image)
        return dict(image=image, y_chain=y_chain, y_country=y_country, y_city=y_city, y_subregion=y_subregion)

    def __len__(self):
        return len(self.dataset)

'''import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from path import Path
import torchvision
from PIL import Image, ImageEnhance, ImageFile
from torchvision import transforms

ROOT_DIR = Path('/hdd2/past_students/virginia/hotel')
DATA_DIR = ROOT_DIR / "data"
#LOCAL_DATA_DIR = Path(__file__).parent.parent/'data'

class Hotelimages(Dataset):
    def __init__(self, split: str):
        assert split in ['train', 'valid', 'test'], split  # così se è un errore mi stampa il valore di split sbagliato
        self.split_ids = pd.read_csv(DATA_DIR / f'{split}.csv')  # f'{}' fa l' interpolazione di stringhe
        self.hotel_info = pd.read_csv(DATA_DIR / 'hotel_info.csv')
        self.images_path = pd.read_csv(DATA_DIR / 'images_path.csv')
        self.dataset = pd.merge(self.split_ids, self.hotel_info, on='hotel_id')
        self.dataset = pd.merge(self.dataset, self.images_path, on='image_id')
        for c in ['chain_id', 'country_id', 'city_id', 'subregion_id']:
            self.dataset[c] = pd.to_numeric(self.dataset[c])
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(self.dataset.columns.tolist())
        print(self.dataset.head())

    def __getitem__(self, index) -> T_co:   #indicizzazione
        image_path = self.dataset.loc[index,'path']
        y = self.dataset.loc[index, ['country_id', 'city_id', 'subregion_id']].astype('int64').values #.values me lo converte in numpy
        y = torch.from_numpy(y)
        image = Image.open(ROOT_DIR/image_path).convert('RGB')
        image = self.preprocess(image)
        return [image, y]

    def __len__(self):
        return len(self.dataset)
'''