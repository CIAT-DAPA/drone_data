
## Crop Monitoring based on drone data



This repository is made to dispose of several drone-based tools for crop monitoring. Currently, there are available examples for:
* Drone data reading
* Spectral indexes calculation
* Plant level 3D visualization 
* Plant level identification given a trained YOLO model
* Cluster classification

## Multitemporal analysis

Considering that crop monitoring involves a continuous capture of data through its cycle. We have implemented a framework in which the data is stored as a multi-dimensional object. Where besides the x and y axis data, a time dimension is included. This refers to when the image was taken. Regarding the spectral bands, those are located 
as a fourth dimensiononal array. Wrapping up, the data is a xarray object with dimensions _time, Spectral band, Y, X_.


### Spectral indexes calculation

You can also calculate different vegetation index layers using the function *.calculate_vi*. To use this function you will need to indicate two parameters:
- vi: which is the name of the vegetation index
- expression: is the equation to calculate the vegetation index, eg. "((green_ms*green_ms) - (red_ms*red_ms))/((green_ms*green_ms) + (red_ms*red_ms))" will calculate the [modified green red vegetation index](https://www.sciencedirect.com/science/article/pii/S0303243415000446).

```python

### Calculando indices vegetales
dronedata.calculate_vi('ndvi')
dronedata.calculate_vi(vi = 'mgrvi', expression = "((green_ms*green_ms) - (red_ms*red_ms))/((green_ms*green_ms) + (red_ms*red_ms))"


```

the following table contains different VI, which can be obtained from combining RGB and NIR spectral bands:

- GRVI: [Red and photographic infrared linear combinations for monitoring vegetation](https://www.sciencedirect.com/science/article/pii/0034425779900130)$$\frac{green - red}{green + red}$$
- MGRVI: [modified green red vegetation index](https://www.sciencedirect.com/science/article/pii/S0303243415000446)$$ \frac{green^2 - red^2}{green^2 + red^2} $$
- RGBVI: [Red Green Blue Vegetation Index](https://www.sciencedirect.com/science/article/pii/S0303243415000446)$$ \frac{(green^2) - (blue*red)}{(green^2) + (blue * red)} $$
- NDVI: [normalized difference vegetation index](https://books.google.co.jp/books?hl=en&lr=&id=e00CAAAAIAAJ&oi=fnd&pg=PA309&ots=JTQteVFm-b&sig=3JNqoOLVGDRe1LNfodW_3T7K9uI&redir_esc=y#v=onepage&q&f=false)$$\frac{nir - red}{red + nir}$$
- NDRE: [normalized difference red edge index](https://www.sciencedirect.com/science/article/pii/S0176161704704034?via%3Dihub)$$\frac{nir - edge}{red + edge}$$
- GNDVI: [green normalized difference vegetation index](https://www.sciencedirect.com/science/article/pii/S0176161704704034?via%3Dihub)$$\frac{nir - green}{red + green}$$
- RECI: [red edge chlorophyll index](https://www.sciencedirect.com/science/article/pii/S0176161704704034?via%3Dihub)$$(\frac{nir}{edge} - 1)$$

the *plotsingleband()* function will a single spectral band

```python
m.plot_singleband('ndvi')

```

<p align="center">

<img src="rm_imgs\multiband.png" alt="rgbimage" id="logo" data-height-percentage="90" data-actual-width="140" data-actual-height="55">


<p align="center">
<img src="https://ciat.cgiar.org/wp-content/uploads/Alliance_logo.png" alt="CIAT" id="logo" data-height-percentage="90" data-actual-width="140" data-actual-height="55">
<img src="https://www.kindpng.com/imgv/hoRRmih_logo-ccafs-hd-png-download/" alt="CCAFS" id="logo2" data-height-percentage="90" width="230" height="52">
</p>

</p>