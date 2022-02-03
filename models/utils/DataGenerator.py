from PIL import Image, ImageFile
import random
import csv
from transformers import GPT2Tokenizer
from torchvision import transforms
import torch
import numpy as np

PARAGRAPH_IMAGES_DIR = '/Volumes/Diego/paragraph_captions/captions/'
COCO_IMAGES_DIR = '/Volumes/Virginia2/coco_dataset/coco_resized/'
PARAGRAPH_CSV_FILE = '/Volumes/Diego/paragraph_captions/all_captions.csv'
COCO_CSV_FILE = 'data/new_coco_captions.csv' #'/Volumes/Virginia2/coco_dataset/new_coco_captions.csv'


class DataGenerator:
    IMG_H = 256
    IMG_W = 256

    def __init__(self, simple=True):

        self.image_caption_dict = dict()

        if simple:
            self.CSV_FILE = COCO_CSV_FILE
            self.IMAGES_DIR = COCO_IMAGES_DIR
            self.gpt_model = 'gpt2'
            self.max_length = 20
        else:
            self.CSV_FILE = PARAGRAPH_CSV_FILE
            self.IMAGES_DIR = PARAGRAPH_IMAGES_DIR
            self.gpt_model = 'gpt2-medium'
            self.max_length = 30

        with open(self.CSV_FILE, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            title = True
            for row in csv_reader:
                if title:
                    title = False
                else:
                    if len(row) == 2:
                        self.image_caption_dict[row[0]] = row[1]  # image_id : caption
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        all_keys = list(self.image_caption_dict.keys())
        random.shuffle(all_keys)
        train_test_split = int(len(all_keys) * 0.9)
        train_valid_split = int(train_test_split * 0.8)
        self.train_keys = all_keys[:train_valid_split]
        self.valid_keys = all_keys[train_valid_split:train_test_split]
        self.test_keys = all_keys[train_test_split:]

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sos_token = '<SOS>'
        self.tokenizer.add_tokens([self.sos_token])

    def get_image(self, image_id):

        loader = transforms.Compose([transforms.Resize((self.IMG_H, self.IMG_W)),  # scale imported image
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image = Image.open(self.IMAGES_DIR + image_id).convert('RGB')  # '.jpg'
        image = loader(image)
        image = image.unsqueeze(0)

        return image

    def generate_batch(self, bs, train=True, no_cuda=False):

        batch_images = []
        batch_y = []

        if train:
            keys = self.train_keys
        else:
            keys = self.valid_keys

        ids = np.random.choice(a=keys, size=bs, replace=True)

        for index, image_id in enumerate(ids):
            print("ID", image_id)
            batch_images.append(self.get_image(str(image_id)))
            caption = self.image_caption_dict[str(image_id)]
            caption = self.sos_token + ' ' + caption

            encoded_caption = self.tokenizer.encode(caption, return_tensors='pt', truncation=True,
                                                    max_length=self.max_length,
                                                    pad_to_max_length=True)
            batch_y.append(encoded_caption)

        b, x, y, z = batch_images[0].shape
        X = torch.Tensor(bs, x, y, z)
        torch.cat(batch_images, out=X)

        y = torch.Tensor(bs, self.max_length)
        y = y.clone().type(dtype=torch.long)
        torch.cat(batch_y, out=y)

        if no_cuda:
            yield X, y
        else:
            yield X.cuda(), y.cuda()
