import pickle

import cv2
from tqdm import tqdm

import numpy as np
import random
import pandas as pd
from PIL import Image, ImageEnhance, ImageFile
import tensorflow as tf
from collections import defaultdict, Counter

from models.utils.attention_visualization import save_attention_over_images
from models.utils.create_geojson import get_country_predictions_geojson

IMAGES_DIR = '/Volumes/Virginia2/Hotel50k/'
HOTEL_INFO_DIR = '/Volumes/Virginia2/floyd_data/data/'

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataGenerator:
    """ This is the superclass for all data generator"""
    # TODO : Each sublcass must implement the following methods : 1. generate_batch, 2. get_target

    IMG_H = 256
    IMG_W = 256

    """ Number of classes """
    COUNTRY_CLASSES = 183
    CITY_CLASSES = 10458
    SUBREGION_CLASSES = 23
    CHAIN_CLASSES = 94

    def __init__(self, bs):
        self.bs = bs

        self.image_paths = {row[0]: row[1] for _, row in
                            pd.read_csv(HOTEL_INFO_DIR + "images_path.csv").iterrows()}  # ImageId : path

        """TRAIN"""
        self.image_hotel_train_dict = {row[0]: row[1] for _, row in
                                       pd.read_csv(HOTEL_INFO_DIR + "train.csv").iterrows()}  # ImageId : HotelId

        """VALID"""
        self.image_hotel_valid_dict = {row[0]: row[1] for _, row in
                                       pd.read_csv(HOTEL_INFO_DIR + "valid.csv").iterrows()}  # ImageId : HotelId

    def rotate(self, img):
        # random selection between rotations
        rotation = random.randrange(-90, 90)
        rotated_img = img.rotate(rotation)
        return rotated_img

    def flip(self, img):
        flip = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
        rotated_img = img.transpose(flip)
        return rotated_img

    def change_brightness(self, img):
        enhancer = ImageEnhance.Brightness(img)

        factor = 1  # gives original image
        enhanced_img = enhancer.enhance(factor)
        return enhanced_img

    def mask(self, img):

        while True:
            y_top = np.random.randint(0, self.IMG_H)  # row index
            x_top = np.random.randint(0, self.IMG_W)  # column index

            height_max = self.IMG_H - y_top
            width_max = self.IMG_W - x_top

            if height_max >= 10 and width_max >= 10:
                break

        height = np.random.randint(5, height_max)
        width = np.random.randint(5, width_max)

        c = np.random.uniform(0, 1)

        img[:, y_top:y_top + height, x_top:x_top + width, :] = c

        return img

    def get_image(self, id, data_augmentation=False):

        path = IMAGES_DIR + self.image_paths[id].split('images/')[1]

        background = Image.open(path).convert('RGB')
        background = background.resize((self.IMG_H, self.IMG_W))

        # if data_augmentation:
        # background = self.mask(background)
        # data_agumentations = [self.rotate, self.flip, self.change_brightness]
        # background = random.choice(data_agumentations)(background)

        img_array = np.array(background)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255.

        return img_array


class HotelDataGenerator(DataGenerator):

    def __init__(self, bs):
        super().__init__(bs)

        ########### ALL HOTEL DATA ###########

        self.hotel_info_dict = {row[0]: [row[1], row[4], row[5], row[6]] for _, row in
                                pd.read_csv(
                                    HOTEL_INFO_DIR + "hotel_info.csv").iterrows()}  # hotel_id : [chain_id, country_id, city_id, subregion_id]

        ########## IMAGES PER COUNTRY - TRAIN ##########

        '''self.images_per_country_train = defaultdict(list)
        for image, hotel in self.image_hotel_train_dict.items():
            country = int(self.hotel_info_dict[int(hotel)][1])
            self.images_per_country_train[country].append(int(image))

        self.countries_train = list(self.images_per_country_train.keys())

        ########## IMAGES PER COUNTRY - VALID ##########

        self.images_per_country_valid = defaultdict(list)
        for image, hotel in self.image_hotel_valid_dict.items():
            country = int(self.hotel_info_dict[int(hotel)][1])
            self.images_per_country_valid[country].append(int(image))

        self.countries_valid = list(self.images_per_country_valid.keys())'''

        ########## IMAGES PER CITY - TRAIN ##########

        self.images_per_city_train = defaultdict(list)
        for image, hotel in self.image_hotel_train_dict.items():
            city = int(self.hotel_info_dict[int(hotel)][1])
            self.images_per_city_train[city].append(int(image))

        self.cities_train = list(self.images_per_city_train.keys())

        ########## IMAGES PER CITY - VALID ##########

        self.images_per_city_valid = defaultdict(list)
        for image, hotel in self.image_hotel_valid_dict.items():
            city = int(self.hotel_info_dict[int(hotel)][1])
            self.images_per_city_valid[city].append(int(image))

        self.cities_valid = list(self.images_per_city_valid.keys())

    def generate_batch(self, train=True):
        while True:
            data_augmentation = np.random.choice((True, False), self.bs, p=[0.5, 0.5])
            batch_images = []
            target_index = 0
            y1 = np.empty(self.bs, dtype=int)
            y2 = np.empty(self.bs, dtype=int)
            y3 = np.empty(self.bs, dtype=int)
            y4 = np.empty(self.bs, dtype=int)

            if train:
                images_per_city = self.images_per_city_train
                cities = self.cities_train
                image_hotel_dict = self.image_hotel_train_dict
            else:
                images_per_city = self.images_per_city_valid
                cities = self.cities_valid
                image_hotel_dict = self.image_hotel_valid_dict

            # TODO : 1. Sample city, 2. sample images, 3. get image path, 4. get image

            city_batch = np.random.choice(a=cities, size=self.bs, replace=True)
            city_frequency = Counter(city_batch)
            for city, freq in city_frequency.items():
                ids = np.random.choice(a=images_per_city[int(city)], size=freq, replace=True)
                for i, id in enumerate(ids):
                    hotel = int(image_hotel_dict[id])
                    batch_images.extend(self.get_image(id, data_augmentation[i]))
                    y1[target_index] = int(self.hotel_info_dict[hotel][1])
                    y2[target_index] = int(self.hotel_info_dict[hotel][2])
                    y3[target_index] = int(self.hotel_info_dict[hotel][3])
                    y4[target_index] = int(self.hotel_info_dict[hotel][0])
                    target_index += 1

            X = np.array(batch_images)
            y1 = np.array(y1)
            y2 = np.array(y2)
            y3 = np.array(y3)
            y4 = np.array(y4)

            yield X, [y1, y2, y3, y4]

    def get_target(self):
        target_dict = defaultdict(list)
        test_images = {row[0]: row[1] for _, row in
                       pd.read_csv(HOTEL_INFO_DIR + "test.csv").iterrows()}  # ImageId : HotelId

        for image_id, hotel_id in test_images.items():
            info = self.hotel_info_dict[hotel_id]
            int_info = [int(i) for i in info]

            target_dict[str(image_id)] = [int_info[1], int_info[2], int_info[3],
                                          int_info[0]]  # country, city, subregion, chain

        return target_dict


############## MODEL ###############
class ParametrisedCompatibility(tf.keras.layers.Layer):

    def __init__(self, kernel_regularizer=None, **kwargs):
        super(ParametrisedCompatibility, self).__init__(**kwargs)
        self.regularizer = kernel_regularizer

    def build(self, input_shape):
        self.u = self.add_weight(name='u', shape=(input_shape[0][3], 1), initializer='uniform', regularizer=self.regularizer, trainable=True)
        super(ParametrisedCompatibility, self).build(input_shape)

    def call(self, x):  # x = [l1,g_l1] add l and g. Dot the sum with u.
        return tf.keras.backend.dot(tf.keras.backend.map_fn(lambda lam: (lam[0]+lam[1]),elems=(x),dtype='float32'), self.u)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])


class LearnToPayAttention:
    IMG_H = 256
    IMG_W = 256

    TRAIN_SAMPLES = 938809
    VALID_SAMPLES = 230143

    COUNTRY_CLASSES = 183
    CITY_CLASSES = 10458
    SUBREGION_CLASSES = 23
    CHAIN_CLASSES = 94

    def __init__(self, bs):
        self.vgg = tf.keras.applications.VGG19(input_shape=(self.IMG_H, self.IMG_W, 3), weights='imagenet',
                                               include_top=False)  # pooling = 'avg'
        counter = 0
        for layer in self.vgg.layers:
            if counter < 18:
                layer.trainable = False
            else:
                layer.trainable = True

        self.bs = bs

    def multi_head_attention(self, li, g_li, li_channels, i, head):

        param_compat_li = ParametrisedCompatibility()
        compatibility_map_li = param_compat_li([li, g_li])  # (batch,x,y,1)
        flatt_i = tf.keras.layers.Flatten(name='flat_att' + str(i) + '_' + str(head))(
            compatibility_map_li)  # (batch, xy)
        attention_map_li = tf.keras.layers.Softmax(name='att_map_' + str(i) + '_' + str(head))(flatt_i)  # (batch, xy)

        xy = attention_map_li.shape[1]

        reshaped_li = tf.keras.layers.Reshape((xy, li_channels), name='reshape_l' + str(i) + '_' + str(head))(li)
        gi = tf.keras.layers.Lambda(lambda lam: tf.keras.backend.squeeze(
            tf.keras.backend.batch_dot(tf.keras.backend.expand_dims(lam[0], 1), lam[1]), 1),
                                    name='g' + str(i) + '_' + str(head))([attention_map_li, reshaped_li])

        return gi, attention_map_li, compatibility_map_li


    def create_model(self):
        x = self.vgg.output
        x = tf.keras.layers.Dropout(0.5)(x)

        g = tf.keras.layers.Flatten()(x)

        local1 = self.vgg.get_layer('block3_conv4')
        local2 = self.vgg.get_layer('block4_conv4')
        local3 = self.vgg.get_layer('block5_conv4')

        # ATTENTION L1
        l1_channels = local1.output.shape[3]
        l1 = local1.output  # (batch,x,y,channels)
        g_l1 = tf.keras.layers.Dense(units=l1_channels)(g)  # (None, channels)

        # 3 HEADED ATTENTION
        g1, attention_map_l1, compatibility_map_l1 = self.multi_head_attention(l1, g_l1, l1_channels, 1, 0)

        # ATTENTION L2
        l2_channels = local2.output.shape[3]
        l2 = local2.output  # (batch,x,y,channels)
        g_l2 = tf.keras.layers.Dense(units=l2_channels)(g)  # (None, channels)

        # 3 HEADED ATTENTION
        g2, attention_map_l2, compatibility_map_l2 = self.multi_head_attention(l2, g_l2, l2_channels, 2, 0)

        # ATTENTION L3
        l3_channels = local3.output.shape[3]
        l3 = local3.output  # (batch,x,y,channels)
        g_l3 = tf.keras.layers.Dense(units=l3_channels)(g)  # (None, channels)

        # 3 HEADED ATTENTION
        g3, attention_map_l3, compatibility_map_l3= self.multi_head_attention(l3, g_l3, l3_channels, 3, 0)

        # OUTPUT LAYER
        g_layers = tf.keras.layers.Concatenate()([g1, g2, g3])  #

        country_softmax = tf.keras.layers.Dense(self.COUNTRY_CLASSES, activation='softmax')(g_layers)
        city_softmax = tf.keras.layers.Dense(self.CITY_CLASSES, activation='softmax')(g_layers)
        subregion_softmax = tf.keras.layers.Dense(self.SUBREGION_CLASSES, activation='softmax')(g_layers)
        chain_softmax = tf.keras.layers.Dense(self.CHAIN_CLASSES, activation='softmax')(g_layers)

        self.model = tf.keras.Model(inputs=self.vgg.input,
                                    outputs=[country_softmax, city_softmax, subregion_softmax, chain_softmax]) #
        self.attention_model_l1 = tf.keras.Model(inputs=self.vgg.input,
                                                   outputs=[attention_map_l1, compatibility_map_l1, g1])
        self.attention_model_l2 = tf.keras.Model(inputs=self.vgg.input,
                                                 outputs=[attention_map_l2, compatibility_map_l2, g2])
        self.attention_model_l3 = tf.keras.Model(inputs=self.vgg.input,
                                                 outputs=[attention_map_l3, compatibility_map_l3, g3])

    def compile_model(self, lr=1e-5, metrics=['accuracy']):

        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.model.compile(optimizer=optimizer, metrics=metrics, loss=loss)

        self.attention_model_l1.compile(optimizer=optimizer, metrics=metrics, loss=loss)
        self.attention_model_l2.compile(optimizer=optimizer, metrics=metrics, loss=loss)
        self.attention_model_l3.compile(optimizer=optimizer, metrics=metrics, loss=loss)

    def build_model(self):
        self.create_model()
        self.compile_model()

    def run_model(self, data_generator, epochs=1):

        train_steps = int(self.TRAIN_SAMPLES / self.bs)
        valid_steps = int(self.VALID_SAMPLES / self.bs)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

        '''self.model.fit(x=data_generator.generate_batch(train=True),
                       epochs=epochs,  #### set repeat in training dataset
                       steps_per_epoch=train_steps)'''

        self.model.fit(x=data_generator.generate_batch(train=True),
                       epochs=epochs,  #### set repeat in training dataset
                       steps_per_epoch=train_steps,
                       validation_data=data_generator.generate_batch(train=False),
                       validation_steps=valid_steps,
                       callbacks=[callback])

    def get_image(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.IMG_H, self.IMG_W))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255.

        return img_array

    def test_model(self, image_path):

        img = self.get_image(image_path)
        predictions = self.model.predict(img)

        p1 = predictions[0][0]  # np.argmax()
        ordered1 = p1.argsort()[::-1][:10]

        p2 = predictions[1][0]
        ordered2 = p2.argsort()[::-1][:10]

        # SUBREGION prediction
        p3 = predictions[2][0]
        ordered3 = p3.argsort()[::-1][:5]

        # CHAIN prediction
        p4 = predictions[2][0]
        ordered4 = p4.argsort()[::-1][:5]

        countries_dict = {row[1]: row[0] for _, row in
                                       pd.read_csv("static/hotel_info/country_id.csv").iterrows()}  # country : id

        get_country_predictions_geojson(countries_dict, p1)

        return [p1,p2,p3,p4, ordered1, ordered2, ordered3, ordered4]

    def get_attention_maps(self, image_path, self_attention_paths):

        img = self.get_image(image_path)

        attention_map_0 = self.attention_model_l1.predict(img)[0][0].reshape((64,64))
        attention_map_1 = self.attention_model_l2.predict(img)[0][0].reshape((32, 32))
        attention_map_2 = self.attention_model_l3.predict(img)[0][0].reshape((16, 16))

        save_attention_over_images(img, attention_map_0, 'static/' + self_attention_paths[0], upscale=64)
        save_attention_over_images(img, attention_map_1, 'static/' + self_attention_paths[1], upscale=32)
        save_attention_over_images(img, attention_map_2, 'static/' + self_attention_paths[2], upscale=16)

        dim = (256, 256)

        avg_attention = np.mean([cv2.resize(attention_map_0, dim, interpolation=cv2.INTER_NEAREST),
                                 cv2.resize(attention_map_1, dim, interpolation=cv2.INTER_NEAREST),
                                 cv2.resize(attention_map_2, dim, interpolation=cv2.INTER_NEAREST)], axis=0)

        return avg_attention

    def get_image(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize((self.IMG_H, self.IMG_W))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255.

        return img_array

    def get_target(self):
        target_dict = defaultdict(list)
        test_images = {row[0]: row[1] for _, row in
                       pd.read_csv(HOTEL_INFO_DIR + "test.csv").iterrows()}  # ImageId : HotelId

        hotel_info_dict = {row[0]: [row[1], row[4], row[5], row[6]] for _, row in
                                pd.read_csv(
                                    HOTEL_INFO_DIR + "hotel_info.csv").iterrows()}  # hotel_id : [chain_id, country_id, city_id, subregion_id]

        for image_id, hotel_id in test_images.items():
            info = hotel_info_dict[hotel_id]
            int_info = [int(i) for i in info]

            target_dict[str(image_id)] = [int_info[1], int_info[2], int_info[3],
                                          int_info[0]]  # country, city, subregion, chain

        return target_dict

    def test_all(self):

        target_dict = self.get_target()


        """ imageID : path"""
        images_path = {row[0]: row[1] for _, row in
                       pd.read_csv(HOTEL_INFO_DIR + "images_path.csv").iterrows()}  # id : path

        total_predictions = 0

        # COUNTRY PREDICTION
        correct_predictions_11 = 0
        correct_predictions_15 = 0  # top 5
        correct_predictions_110 = 0  # top 10

        # CITY PREDICTION
        correct_predictions_21 = 0
        correct_predictions_25 = 0  # top 5
        correct_predictions_210 = 0  # top 10

        # SUBREGION PREDICTION
        correct_predictions_31 = 0
        correct_predictions_35 = 0  # top 5
        correct_predictions_310 = 0

        # CHAIN PREDICTION
        correct_predictions_41 = 0
        correct_predictions_45 = 0  # top 5
        correct_predictions_410 = 0

        for filename, target in tqdm(target_dict.items()):
            filename = int(float(filename))
            image_path = images_path[filename][7:]
            img = self.get_image(IMAGES_DIR + image_path)
            predictions = self.model.predict(img)

            # COUNTRY prediction
            p1 = predictions[0][0]  # np.argmax()
            ordered1 = p1.argsort()[::-1][:10]  # TAKE FIRST 10 PREDICTED COUNTRIES
            p1_1 = ordered1[0]

            # CITY prediction
            p2 = predictions[1][0]
            ordered2 = p2.argsort()[::-1][:10]
            p2_1 = ordered2[0]

            # SUBREGION prediction
            p3 = predictions[2][0]
            ordered3 = p3.argsort()[::-1][:3]
            p3_1 = ordered3[0]

            # CHAIN prediction
            p4 = predictions[3][0]
            ordered4 = p4.argsort()[::-1][:3]
            p4_1 = ordered4[0]

            # Targets
            t1 = int(target[0])
            t2 = int(target[1])
            t3 = int(target[2])
            t4 = int(target[3])

            if t1 == p1_1:
                correct_predictions_11 += 1
                print("image", filename)
                print("country", t1)
            if t1 in ordered1[:5]:
                correct_predictions_15 += 1
            if t1 in ordered1[:10]:
                correct_predictions_110 += 1

            if t2 == p2_1:
                correct_predictions_21 += 1
                print("image", filename)
                print("city", t2)
            if t2 in ordered2[:5]:
                correct_predictions_25 += 1
            if t2 in ordered2[:10]:
                correct_predictions_210 += 1

            if t3 == p3_1:
                correct_predictions_31 += 1
                print("image", filename)
                print("subregion", t3)
            if t3 in ordered3[:5]:
                correct_predictions_35 += 1
            if t3 in ordered3[:10]:
                correct_predictions_310 += 1

            if t4 == p4_1:
                correct_predictions_41 += 1
            if t4 in ordered4[:5]:
                correct_predictions_45 += 1
            if t4 in ordered4[:10]:
                correct_predictions_410 += 1

            total_predictions += 1

        print("Total predictions", total_predictions)
        print("Correct country at first guess", correct_predictions_11)
        print("Percentage: %d %%" % (correct_predictions_11 * 100 / total_predictions))
        print("Country in top 5", correct_predictions_15)
        print("Percentage: %d %%" % (correct_predictions_15 * 100 / total_predictions))
        print("Country in top 10", correct_predictions_110)
        print("Percentage: %d %%" % (correct_predictions_110 * 100 / total_predictions))

        print("Correct city at first guess", correct_predictions_21)
        print("Percentage: %d %%" % (correct_predictions_21 * 100 / total_predictions))
        print("City in top 5", correct_predictions_25)
        print("Percentage: %d %%" % (correct_predictions_25 * 100 / total_predictions))
        print("City in top 10", correct_predictions_210)
        print("Percentage: %d %%" % (correct_predictions_210 * 100 / total_predictions))

        print("Correct subregion at first guess", correct_predictions_31)
        print("Percentage: %d %%" % (correct_predictions_31 * 100 / total_predictions))
        print("Subregion in top 5", correct_predictions_15)
        print("Percentage: %d %%" % (correct_predictions_35 * 100 / total_predictions))
        print("Subregion in top 10", correct_predictions_310)
        print("Percentage: %d %%" % (correct_predictions_310 * 100 / total_predictions))

        print("Correct chain at first guess", correct_predictions_41)
        print("Percentage: %d %%" % (correct_predictions_41 * 100 / total_predictions))
        print("Chain in top 5", correct_predictions_45)
        print("Percentage: %d %%" % (correct_predictions_45 * 100 / total_predictions))
        print("Chain in top 10", correct_predictions_410)
        print("Percentage: %d %%" % (correct_predictions_410 * 100 / total_predictions))

        return correct_predictions_11 / total_predictions


if __name__ == "__main__":
    BS = 16
    geolocation_model = LearnToPayAttention(bs=BS)
    geolocation_model.build_model()
    geolocation_model.model.set_weights(pickle.load(open("weights/gm2h_city_8ep_1e5.pkl", "rb")))
    geolocation_model.test_all()
