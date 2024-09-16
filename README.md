
# Crop Monitoring Using Drone Data

Welcome to our repository dedicated to advanced crop monitoring using drone-acquired data. This project leverages the power of UAV technology to provide insightful agricultural data. Our toolkit includes a range of features tailored for precision agriculture 🌱, designed to enhance crop management and yield optimization.

## 🛠️ Key Features

This repository is made to dispose of several drone-based tools for crop monitoring. Currently, there are available examples for:

* UAV Orthomosaic Clipping 🗺️: Generate subset of orthomosaic images using spatial boundaries.
* Spectral Indices Calculation 📊: Calculate various spectral indices for assessing plant health. This tool supports indices like NDVI, NDRE, and more, offering a deep insight into crop vitality.
* 3D Visualization 🌐: Transform your data into interactive 3D models. Visualize crop height.
* Plant detection given with YOLOv5🌿: Deploy a pre-trained YOLO (You Only Look Once) model for accurate plant detection. 
* Cluster classification 🔍: Apply advanced clustering algorithms for segmenting crops based on various criteria, facilitating targeted interventions and analysis.

## 📈 Multi-temporal Analysis Framework

Our framework is designed for efficient handling of crop monitoring data across different growth stages. It features:

* Structured Data Storage 💾: We employ a multi-dimensional approach to data organization. Our framework combines spatial (X, Y), temporal (time), and spectral (spectral band) dimensions into an integrated xarray object. This structure facilitates easy access and manipulation of complex datasets.
* Time-Series Analysis ⏳: Track and analyze changes in crop health and growth over time. Our time dimension allows for seamless comparison of data across different dates.