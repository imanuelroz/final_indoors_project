# A Hierarchical Geolocation of Indoor Scenes with Visual and Text Explanations
A pipeline for predicting the location of an indoor scene at three hierarchical levels, namely city, country and subregion level, and for providing different visual explanations.

## About
This repository contains the work I developed together with prof. Mark J. Carman for my master thesis in Mathetamical Engineering at Politecnico di Milano.

The pipeline is made of 2 components:
* a hierarchical geolocating model (all the tested models are inside the models directory, the best performanece were achieved using swin_b)
* a network for segmentation to be combined with SHAP algorithm in order to produce meaningful visual explanations. 
* (in addition we tested other XAI techniques as: Integrated Gradients and Grad-CAM)


## Running the demo
To run the demo and try out the pipeline do the following
1. On your terminal run: ssh -L 7860:127.0.0.1:7860 usernabame@server_address
2. Once connected run bash command 


### Training procedure
1. Open your terminal and run the train.py file choosing the pretrained backbone you want to use for training.

