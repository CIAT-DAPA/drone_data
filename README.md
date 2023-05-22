
## Crop Monitoring based on drone data

This repository is made to dispose of several drone-based tools for crop monitoring. Currently, there are available examples for:
* UAV orthomosaic cliping
* Spectral indices calculation
* 3D visualization 
* Plant detection given a trained YOLO model
* Cluster classification

## Multi-temporal analysis

In order to facilitate the management and analysis of crop monitoring data throughout the growth cycle, we have developed a comprehensive framework for data storage and organization. In this framework, the data is structured as a multi-dimensional object, incorporating not only the x and y spatial dimensions but also a time dimension, indicating the specific time point when the images were captured. Additionally, the spectral bands are represented as a fourth-dimensional array within the data structure. In summary, the data is stored as an xarray object with the following dimensions: time, Spectral band, Y, X. This structured representation allows for efficient data handling, exploration, and analysis in the context of crop monitoring.