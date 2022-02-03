import pandas as pd
from PIL import Image
from path import Path


ROOT_DIR = Path('/Volumes/LACIE\ SHARE/hotel/')
DATA_DIR = Path('/Volumes/LACIE\ SHARE/hotel/data/')

def main():
    images_path = pd.read_csv(DATA_DIR / 'images_path.csv')
    for i in images_path.index:
        img_path = images_path.loc[i, 'path']  # vedi se mettere loc
        img = Image.open(ROOT_DIR + '/' + img_path)
        img_type = img.getbands()  # stringa con tipo
        filtering = 'greyscale' if img_type == "L" else 'RGB'
        images_path.at[i, 'filter'] = filtering
    images_path = images_path.loc[images_path['filter'] == 'RGB']
    images_path.to_csv(DATA_DIR/'images_path_RGB_only.csv', index=False)

if __name__ == '__main__':
    main()