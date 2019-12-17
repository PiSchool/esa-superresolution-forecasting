<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ6ELxXcZzhZlcyKtNAYf4woGljLbxPKHRJUyTbM_bVlPrWQ_9b&s" width="150" height="40">

<img src="http://www.spacetechexpo.eu/assets/files/images/news%20pages/BRE/esa_logo2.jpg" width="95" height="55"> |
<img src="http://www.meeo.it/wp/wp-content/uploads/2014/01/meeo_logo_trans.png" width="95" height="55"> | 
<img src="http://www.sistema.at/wp/wp-content/uploads/2017/10/LOGO_def_sistema.png" width="95" height="55"> | 
<img src="https://c.cs85.content.force.com/servlet/servlet.ImageServer?id=0156E000000Kg6fQAC&oid=00D6E000000DZCb" width="95" height="55"> | 
<img src="http://www.spaceexe.com/wp-content/uploads/2019/09/UrbyetOrbit_space.png" width="95" height="55">


# Sentinel-5p forecasting

This repository includes the PyTorch implementation of an encoder-decoder forecasting network. It was built for the prediction of air-pollution variables based on Sentinel-5p imagery.

## Introduction

Earth observation is producing a large amount of data for multiple applications as agriculture, land management, maritime surveillance, meteorological prediction… In the context of climate change the analysis of the evolution of atmospheric pollution is more and more needed.

With the context of big data in earth observation and the development of accurate methods in Artificial Intelligence there is an interest growing for combining both fields and to provide better analysis and data. 

## Problem statement

In atmospheric pollution, a numerical model produced by the Copernicus Atmosphere Monitoring Service (CAMS) is commonly used. The data from this model is produced by combination of ground measurement and satellite data to monitor pollution variable in the total column of the atmosphere. This data is at a global scale with a 40km spatial resolution and a hourly temporal resolution. Moreover the numerical model is providing 5 days of forecasting of the pollution variables.

Satellite data are more and more used, and due to it resolution the interest in it is growing up. Sentinel-5p is one of this satellite, launched in 2017 by the European Space Agency. It perform measurements of pollution variable concentration in the atmosphere, all around the world at a 5 km spatial resolution and with daily revisit time.

The main objective of earth observation field is to get better resolution data, in terms of spatial and temporal resolution. In this context of optimisation of the data, the main idea of the work presented here is to reach 5 days of forecasting of pollution variables at the same resolution than Sentinel-5p. For this project the focus was done on NO2 concentration in the atmosphere.

## Solution

The solution proposed in the respository consist in the prediction of the next Sentinel-5p image. The model learnt on temporal sequences to detect how the images are evolving through time. By considering the last image available the model is able to provide five days of forecasting of NO2 concentration in the atmosphere with one image predicted per day.

<img src="https://github.com/MaxHouel/First/blob/master/random/prediction_solution.PNG?raw=true" width="1000" height="125">

## Model architecture

## Results

## Setup to get started
Make sure you have Python3 installed.
 You can install the required python packages by running:
```console
pip install -r requirements.txt
```

Before starting the model, place your configuration in the `config.json` file.
You can configure the following parameters:
```
name: name of the folder in which logs and model checkpoints are saved.
n_gpu: number of gpus for training, multi-gpu training is supported.
arch: model parameters
data_loader: options for the loading of the dataset
optimizer: optimization type and learning rate
loss: loss function used for training
metrics: metrics used for evaluation
trainer: training specifications, such as number of epochs and early stopping 
```

## Usage:

### Training:
```console
python main.py -c config.json
```

Using multiple GPU:
```console
python main.py --device 0,1,2,3 -c config.json
```

### Evaluation
```console
python test.py -c config.json -r /path/to/model_checkpoint
```

