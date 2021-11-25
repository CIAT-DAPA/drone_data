
## Procesar imágenes de drones
<p align="center">
<img src="https://ciat.cgiar.org/wp-content/uploads/Alliance_logo.png" alt="CIAT" id="logo" data-height-percentage="90" data-actual-width="140" data-actual-height="55">
<img src="https://www.kindpng.com/imgv/hoRRmih_logo-ccafs-hd-png-download/" alt="CCAFS" id="logo2" data-height-percentage="90" width="230" height="52">
</p>


Este repositorio se creó con la intención de facilitar el procesamiento de imágenes de drones, con la intención de crear fácilmente índices vegetales y mapas de coberturas, a partir de bandas espectrales tomadas por drones.
A continuación se muestra un ejemplo de su uso y de algunas funciones disponibles hasta el momento.

Lo primero es señalar la carpeta en donde se encuentran ubicadas las imágenes de drones.
De  igual forma, se debe indicar el bnombre de las bandas a leer. *Actualmente, solo se puede leer una imágen for fecha.*

La función drone_data.DroneData, almacena la información en una matriz de tres dimensiones, x * y * banda.

```python
from utils import drone_data
from utils import plt_functions
import numpy as np

m = drone_data.DroneData("images/", ## directorio donde se encuentran las imágenes
                         bands=['blue', 'green', 'red', 'r_edge', 'nir'] ## nombre de las bandas de las imagenes
                         )
```
### Añadir indices vegetales

Con la función *.calculate_vi*, se pueden realizar distintas operaciones entre bandas, como defecto se encuentra las bandas ndvi y ndvire.
``
<br>
$$ndvi = \frac{nir - red}{red + nir}$$
<br>
$$ndvire = \frac{nir - rededge}{rededge + nir}$$

También es posible agregar otros tipo de indices vegetales, expresando la función, ejemplo: "(nir-green)/(nir + green)"
```python

### Calculando indices vegetales
m.calculate_vi('ndvi')
m.calculate_vi('ndvire')
```

La función *plotsingleband()*, permite mostrar una banda almacenada en el cubo de datos.
```python
m.plot_singleband('ndvi')

```
De igual forma se cuenta con la función *plot_multiplebands* para poder crear mapas combinando distintas bandas.

```python
m.plot_multiplebands(['red', 'green', 'blue'])

```
<p align="center">

<img src="rm_imgs\multiband.png" alt="rgbimage" id="logo" data-height-percentage="90" data-actual-width="140" data-actual-height="55">

</p>