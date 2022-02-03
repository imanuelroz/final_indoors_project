import pandas as pd
from PIL import Image
from path import Path
import torch
from torchvision import transforms

from datasets.hotel50k import DATA_DIR, ROOT_DIR

dest_folder = Path(__file__).parent/'data'

def main():
    images_path = pd.read_csv(DATA_DIR / 'images_path.csv')
    for i in images_path.index:
        img_path = images_path.loc[i, 'path']  # vedi se mettere loc
        img = Image.open(ROOT_DIR + '/' + img_path)
        convert_tensor = transforms.ToTensor()
        conv_img = convert_tensor(img)
        img_type = img.getbands()  # stringa con tipo
        filtering = 'greyscale' if conv_img.shape[0] == 1 else 'RGB'
        images_path.at[i, 'filter'] = filtering
    images_path = images_path.loc[images_path['filter'] == 'RGB']
    images_path.to_csv(dest_folder/'images_path_rgb_only.csv', index=False)

if __name__ == '__main__':
    main()
