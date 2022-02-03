# A Hierarchical Geolocation of Indoor Scenes with Visual and Text Explanations
A pipeline for predicting the location of an indoor scene at three hierarchical levels, namely city, country and subregion level, and for providing both visual and text explanations.

## About
This repository contains the work I developed together with prof. Mark J. Carman for my master thesis in Computer Science and Engineering at Politecnico di Milano.
This work is a tool born to be used in specific applications such as law enforcement to narrow down the search of a location given a picture of an indoor scene and to point out which features have been the most relevant for the prediction, both in a visual and textual manner.

The pipeline is made of 3 components:
* a hierarchical geolocating model
* self-attention maps and a pre-trained object detector (DETR) to provide visual explanations
* a captioning model to generate a text description of the scene


## Running the demo
To run the demo and try out the pipeline do the following
1. On your terminal go the the repository directory
2. Run the following
```
python home.py
```
3. Open the prompted local address to launch the demo on your browser

## Training the Geolocation Model
### Requirements
* Docker
### Training procedure
1. Open your terminal and go to the training directory in your local git copy of this repository
2. Run the following:
```
docker run -it --runtime=nvidia -v $(pwd):/hotel hotel_geolocation 
python hotel/run.py
```
If you want to change the geolocation model (among GM1H, GM2H, GM3H) and other training parameters edit the `run.py` script

