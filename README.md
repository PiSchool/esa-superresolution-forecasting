# Sentinel-5p forecasting

This repository includes the PyTorch implementation of an encoder-decoder forecasting network. It was built for the prediction of air-pollution variables based on two ESA data products: 
Atmospheric measurements sensed by the [Sentinel-5p](https://sentinel.esa.int/web/sentinel/missions/sentinel-5p) satellite and an air-quality forecasting model provided by the [Copernicus Atmospheric Monitoring Service](https://atmosphere.copernicus.eu/data) that combines satellite observations with sophisticated chemistry and transport models.
<!--- 
## Introduction

The Copernicus program is a European program for Earth Observation and monitoring. The Sentinel mission from the European Space Agency is part of this initiative, it consists of six missions of different satellite sensors for multiple applications as land management, maritime surveillance, atmosphere monitoring… 

Air pollution is known as one of the biggest risks for the environment and human health, due to it consequences the need to monitor it is growing.

Moreover the context of big data in earth observation and the constant development of accurate methods in Artificial Intelligence is creating new opportunity for remote sensing community. By applying deep learning methods on satellite imagery, it produce better data and better analysis.

## Problem statement

In atmospheric pollution, a numerical model produced by the Copernicus Atmosphere Monitoring Service (CAMS) is commonly used. The data produced is a combination of ground measurement and satellite data to monitor the concentration of gases in the total column of the atmosphere. This data is at a global scale with a 40km spatial resolution and an hourly temporal resolution. Moreover the numerical model is providing 5 days of forecasting of the pollution variables.
Due to the resolution of satellite data, the interest in it is growing up. Sentinel-5p is one of this satellite, launched in 2017 by the European Space Agency. It performs measurements every day all around the world for mapping the pollution gases in the atmosphere with a spatial resolution of 5 km.

## Solution

In this project to provide better information in air pollution monitoring, based on the data available for forecasting of pollution variable, we decided to produce a better resolution data for this purpose. Based on deep learning techniques, the Convolution Neural Network (CNN) learnt from temporal sequences to detect the evolution of Sentinel-5p images through time. By considering the last available image, the model is able to provide five days of forecasting of Sentinel-5p for NO2 concentration in the atmosphere.

<img src="https://github.com/MaxHouel/First/blob/master/random/prediction_solution.PNG?raw=true" width="1000" height="150">
-->
## Model architecture

The encoder-decoder network uses a [Convolutional LSTM](https://arxiv.org/abs/1506.04214) architecture. The spatial structure of the input is preserved throught the layers of the network. 
The model is trained using a masked MSE loss and the ADAM optimizer.

We employ two different versions of this architecture. One uses Sentinel-5 sequences solely (s5-fc), the other creates a prediction using the numerical forecast data as a conditional input to the decoder (Cond-S5-fc).


## Results

The model's performance is evaluated on the Peak signal-to-noise ratio (PSNR) and Structural Similarity Index Measure (SSIM).

| Model     | PSNR   | SSIM   |
|-----------|:------:|-------:|
| S5-fc     | 21.08  | 0.52   |
|Cond-S5-Fc | 31.05  | 0.70   |

Both models were evaluated on a test set of 300 sequences with 5 frames being used as input and the following 5 frames being predicted by the network. 

![alt text]( https://github.com/PiSchool/esa-superres-github/blob/master/data/trainloss.png "Training loss")

![alt text]( https://github.com/PiSchool/esa-superres-github/blob/master/data/val_acc.png "Validation accuracy")

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

## License

This project is licensed under the MIT License. See LICENSE for more details.
Except as contained in this notice, the name of the authors shall not be used in advertising or otherwise to promote the sale, use or other dealings in this Software without prior written authorization from the authors.

## Acknowledgements

This work is the result of a challenge proposed by ESA as part of the Pi School of AI 2019 Q4 programme.
We are grateful to all organizers, stakeholders and mentors for providing us this opportunity.



<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ6ELxXcZzhZlcyKtNAYf4woGljLbxPKHRJUyTbM_bVlPrWQ_9b&s" width="150" height="40">

<img src="http://www.spacetechexpo.eu/assets/files/images/news%20pages/BRE/esa_logo2.jpg" width="95" height="55"> |
<img src="http://www.meeo.it/wp/wp-content/uploads/2014/01/meeo_logo_trans.png" width="95" height="55"> | 
<img src="http://www.sistema.at/wp/wp-content/uploads/2017/10/LOGO_def_sistema.png" width="95" height="55"> | 
<img src="https://c.cs85.content.force.com/servlet/servlet.ImageServer?id=0156E000000Kg6fQAC&oid=00D6E000000DZCb" width="95" height="55"> | 
<img src="http://www.spaceexe.com/wp-content/uploads/2019/09/UrbyetOrbit_space.png" width="95" height="55">


