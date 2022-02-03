import csv
import os
import matplotlib
import io
import base64

matplotlib.use("TKAgg")
import skimage.transform
import matplotlib.pyplot as plt
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pickle
import pandas as pd
from skimage.transform.pyramids import _smooth

from models.utils.DataGenerator import DataGenerator
from models.utils.evaluator import Evaluator, load_checkpoint
from models.utils.CoAttention import CoAttention
from models.utils.VGGEncoder import VGGEncoder
#from models.main_models.M1 import AttentionModel
#from models.main_models.M2 import Model
from models.main_models.M3 import Model
#from models.main_models.M4 import AttentionModel
#from models.main_models.M5 import Model
#from models.main_models.M6 import Model


#from models.main_models.GM1H import LearnToPayAttention
from models.main_models.GM2H import LearnToPayAttention
#from models.main_models.GM3H import LearnToPayAttention

from models.utils.object_detection import ObjectDetector

import numpy as np
from PIL import Image
from torchvision import transforms
import random

app = Flask(__name__)

# Uploads Config
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Model Config
#random.seed(1234)
evaluator = Evaluator()

data_generator_simple = DataGenerator(simple=True)
#data_generator_paragraph = DataGenerator(simple=False)

#model_m1 = load_checkpoint('models/pretrained_captioning/m1.pth')
#model_m2 = load_checkpoint('models/pretrained_captioning/m2.pth')
#model_m3 = load_checkpoint('models/pretrained_captioning/m3.pth')
#model_m4 = load_checkpoint('models/pretrained_captioning/m4.pth')
#model_m5 = load_checkpoint('models/pretrained_captioning/m5.pth')
#model_m6 = load_checkpoint('models/pretrained_captioning/m6.pth')


#SELECTED_MODEL = model_m3
SELECTED_DATA_GEN = data_generator_simple


loader = transforms.Compose([transforms.Resize((256, 256)),  # scale imported image
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#all_images = ['airbnb/' + file for file in os.listdir('static/airbnb')] + ['hotel50k/' + file for file in
                                                                           #os.listdir('static/hotel50k')]
#random.shuffle(all_images)
#GALLERY_ITEMS = random.sample(all_images, 15)

BS = 16
geolocation_model = LearnToPayAttention(bs=BS)
geolocation_model.build_model()
#geolocation_model.model.set_weights(pickle.load(open("models/pretrained_geolocation/gm2h_city_8ep_1e5.pkl", "rb")))
#geolocation_model.model.set_weights(pickle.load(open("models/pretrained_geolocation/gm2h.pkl", "rb")))

object_detector = ObjectDetector()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/demo')
def loadDemo():
    return render_template('demo.html', items=GALLERY_ITEMS)

@app.route('/data_visualization')
def load_data_visualization():
    return render_template('data_visualization.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed')
        return render_template('demo.html', folder='uploads', filename=filename, items=GALLERY_ITEMS)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<folder>/<filename>')
def display_image(folder, filename):
    return redirect(url_for('static', filename=folder + '/' + filename), code=301)

@app.route('/focused_caption/<folder>/<filename>', methods=['POST'])
def generate_focused_caption(folder, filename):
    # print('display_image filename: ' + filename)
    filepath = 'static/' + folder + '/' + filename

    self_attention_paths = ['tmp/self_attention_0.png', 'tmp/self_attention_1.png', 'tmp/self_attention_2.png']
    avg_attention = geolocation_model.get_attention_maps(filepath, self_attention_paths)

    object_detector.mask_img(filepath, avg_attention)

    masked_path = 'static/tmp/masked.png'

    X = Image.open(masked_path).convert('RGB')
    original = X.resize((256, 256))
    X = loader(X)
    X = X.unsqueeze(0)

    result, softmaps_list = evaluator.evaluate(X, SELECTED_DATA_GEN, SELECTED_MODEL)

    attentions = []

    new_maps = [softmap.view(7, 7) for softmap in softmaps_list]
    word_list = result.split('  ')
    plt.ioff()
    plt.clf()
    plt.figure(frameon=False)
    for i in range(len(word_list)):
        if word_list[i] == '|<end_of_text>|' or word_list[i] == '':
            break

        img = io.BytesIO()

        alpha = skimage.transform.pyramid_expand(new_maps[i].detach().numpy(), upscale=224 / 7, sigma=8)
        plt.text(0, 0, '%s' % word_list[i], color='white', backgroundcolor='black', fontsize=30)
        plt.imshow(original)
        plt.imshow(alpha, alpha=0.7)

        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        attentions.append(plot_url)

        plt.clf()

    return render_template('demo.html', folder=folder, filename=filename, items=GALLERY_ITEMS,
                           captions=[result], attentions=attentions)


@app.route('/caption/<folder>/<filename>', methods=['POST'])
def generate_caption(folder, filename):
    # print('display_image filename: ' + filename)
    filepath = 'static/' + folder + '/' + filename

    X = Image.open(filepath).convert('RGB')
    original = X.resize((256, 256))
    X = loader(X)
    X = X.unsqueeze(0)

    result, softmaps_list = evaluator.evaluate(X, SELECTED_DATA_GEN, SELECTED_MODEL)

    attentions = []

    new_maps = [softmap.view(7, 7) for softmap in softmaps_list]
    word_list = result.split('  ')
    plt.ioff()
    plt.clf()
    plt.figure(frameon=False)
    for i in range(len(word_list)):
        if word_list[i] == '|<end_of_text>|' or word_list[i] == '':
            break

        img = io.BytesIO()

        alpha = skimage.transform.pyramid_expand(new_maps[i].detach().numpy(), upscale=224 / 7, sigma=8)
        plt.text(0, 0, '%s' % word_list[i], color='white', backgroundcolor='black', fontsize=30)
        plt.imshow(original)
        plt.imshow(alpha, alpha=0.7)

        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        attentions.append(plot_url)

        plt.clf()

    return render_template('demo.html', folder=folder, filename=filename, items=GALLERY_ITEMS,
                           captions=[result], attentions=attentions)


@app.route('/imgClick/<folder>/<filename>')
def img_click(folder, filename):
    return render_template('demo.html', folder=folder, filename=filename, items=GALLERY_ITEMS)


@app.route('/visual_explanations/<folder>/<filename>', methods=['POST'])
def generate_visual_explanations(folder, filename):
    img_path = 'static/'+folder+'/'+filename
    detected_img = object_detector.get_detections(img_path=img_path) #detected_img

    img = io.BytesIO()
    detected_img.save(img, format="PNG")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Self Attention Maps
    self_attention_paths = ['tmp/self_attention_0.png', 'tmp/self_attention_1.png', 'tmp/self_attention_2.png']
    avg_attention = geolocation_model.get_attention_maps(img_path, self_attention_paths)

    original = Image.open(img_path).convert('RGB')
    original = original.resize((256,256))
    img_array = np.array(original)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array / 255.
    plt.ioff()
    plt.clf()

    plt.figure()
    plt.imshow(img_array[0])
    alpha = _smooth(np.array(avg_attention), sigma=8, cval=0, mode='reflect')
    plt.imshow(img_array[0])
    plt.imshow(alpha, alpha=0.7)
    plt.gcf().savefig('static/tmp/avg_attention.png')

    return render_template('demo.html', folder=folder, filename=filename, items=GALLERY_ITEMS, detected_img=plot_url, self_attentions=self_attention_paths)

def get_coords(p2, ordered2):
    coords = []

    city_dict = {row[0]: row[1] for _, row in
                 pd.read_csv("static/hotel_info/city_id.csv").iterrows()}

    coords_dict = {int(row[0]): [row[1],row[2]] for _, row in
                 pd.read_csv("static/hotel_info/city_coords.csv").iterrows()}

    '''#ordered2 = [0,1,2,3,4]
    #p2 = [0.7, 0.2, 0.05, 0.02, 0.01]

    with open("static/hotel_info/city_coords.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] != 'city_id' and int(row[0]) in ordered2:
                idx = ordered2.index(int(row[0]))
                coords.append([str(city_dict[int(row[0])]), float(row[1]), float(row[2]), float(p2[idx])])

    print(coords)'''

    for i, city_id in enumerate(ordered2):
        coordinates = coords_dict[int(city_id)]
        coords.append([str(city_dict[int(city_id)]), coordinates[0], coordinates[1], p2[i]])

    return coords

def get_subregions(p3, ordered3):
    subregion_dict = {int(row[1]): str(row[0]) for _, row in
                 pd.read_csv("static/hotel_info/subregion_id.csv").iterrows()}

    subregions = []

    for i, subregion_id in enumerate(ordered3):
        subregions.append([str(subregion_dict[int(subregion_id)]), p3[i]])

    return subregions


@app.route('/geolocation/<folder>/<filename>', methods=['POST'])
def geolocate(folder, filename):
    img_path = 'static/' + folder + '/' + filename

    p1, p2, p3, p4, ordered1, ordered2, ordered3, ordered4 = geolocation_model.test_model(img_path)

    coords = get_coords(p2, ordered2)

    subregions = get_subregions(p3, ordered3)

    #lat = -27.4698
    #long = 153.0251
    zoom = 2

    return render_template('demo.html', folder=folder, filename=filename, items=GALLERY_ITEMS, geolocation=coords, subregions=subregions)


@app.route('/get_countries_json')
def get_countries_json():
    return redirect(url_for('static', filename='tmp/predictions.geojson', code=301))


@app.route('/crowdsourcing')
def loadCrowdSourcing(filename=None, folder='uploads'):
    cities = []
    countries = []
    subregions = []

    with open('static/hotel_info/city_id.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:
                cities.append(row[1])
    with open('static/hotel_info/country_id.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:
                countries.append(row[1])
    with open('static/hotel_info/subregion_id.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:
                subregions.append(row[0])

    cities.sort()
    countries.sort()
    subregions.sort()

    if filename:
        return render_template('contribute.html', subregions=subregions, countries=countries, cities=cities, filename=filename, folder=folder)
    else:
        return render_template('contribute.html', subregions=subregions, countries=countries, cities=cities)

@app.route('/upload_contribute', methods=['POST'])
def upload_image_contribute():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed')
        return loadCrowdSourcing(folder='uploads', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/handle_contribution/<filename>', methods=['POST'])
def handle_contribution(filename):
    cities_dict = {row[1]: row[0] for _, row in
                        pd.read_csv("static/hotel_info/city_id.csv").iterrows()}  # city : id
    countries_dict = {row[1]: row[0] for _, row in
                   pd.read_csv("static/hotel_info/country_id.csv").iterrows()}  # country : id
    subregion_dict = {row[0]: row[1] for _, row in
                   pd.read_csv("static/hotel_info/subregion_id.csv").iterrows()}  # subregion : id

    country = request.form['country']
    city = request.form['city']
    subregion = request.form['subregion']
    caption = request.form['caption']

    with open('static/contributions.csv','a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(filename), int(countries_dict[country]), int(cities_dict[city]), int(subregion_dict[subregion]), str(caption)])

    return render_template('contribute.html')

if __name__ == '__main__':
    # app.add_url_rule('/', '/hello', 'hello_world')
    app.run()
