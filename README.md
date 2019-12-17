# Sentinel-5p forecasting

This repository includes the PyTorch implementation of an encoder-decoder forecasting network. It was built for the prediction of air-pollution variables based on Sentinel-5p imagery.

![ESA_logo]()
![MEEO Logo]() 
![SISTEMA Logo]() 
![e-Geos Logo]()
![urbyetorbyt Logo]()


## Introduction

Earth observation is producing a large amount of data for multiple applications as agriculture, land management, maritime surveillance, meteorological prediction… In the context of climate change the analysis of the evolution of atmospheric pollution is more and more needed.

With the context of big data in earth observation and the development of accurate methods in Artificial Intelligence there is an interest growing for combining both fields and to provide better analysis and data. 

## Problem statement

In atmospheric pollution, a numerical model produced by the Copernicus Atmosphere Monitoring Service is commonly used. The data from this model is produced by combination of ground measurement and satellite data to monitor pollution variable in the total column of the atmosphere at a global scale with a 40km spatial resolution and a hourly temporal resolution. Moreover the numerical model is providing 5 days of forecasting of the pollution variables.

Satellite data are more and more used, and due to it resolution the interest in it is growing up. Sentinel-5p is one of this satellite, launched in 2017 by the European Space Agency, it perform measurements of pollution variable all around the world at a 5.5km spatial resolution and daily revisit.

The main objective of earth observation field is to get better resolution data, in terms of spatial and temporal resolution. In this context of optimisation of the data, the main idea of the project presented here is to reach 5 days of pollution variables forecasting at the same resolution of Sentinel-5p. 

A better resolution for forecasting, many applications as wildfire / volcanic monitoring, public awereness, will be more accurate than it is already.

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


