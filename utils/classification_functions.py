from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import random
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import numpy as np
from utils import data_processing


def pca_transform(data,
                  variancemin=0.5,
                  export_pca=False):
    """

    :param data: numpy array
    :param varianzemin: numeric
    :param export_pca: boolean
    :return: dictionary
    """

    pca = PCA()
    pca.fit_transform(data)
    # define the number of components through
    ncomponets = np.max(np.argwhere((pca.explained_variance_ * 100) > variancemin)) + 1

    print("calculating pca with {} components".format(ncomponets))
    # calculate new pca
    pca = PCA(n_components=ncomponets).fit(data)
    scaleddata = pca.transform(data)
    # export data
    output = {'pca_transformed': scaleddata}

    if export_pca:
        output['pca_model'] = pca

    return output


def kmeans_images(data, nclusters,
                  scale="minmax",
                  nrndsample="all",
                  seed=123,
                  pca=True,
                  export_pca=False,
                  eigmin=0.3):
    """

    :param data:
    :param nclusters:
    :param scale:
    :param nrndsample:
    :param seed:
    :param pca:
    :param export_pca:
    :param eigmin:
    :return:
    """
    if scale == "minmax":
        scaler = MinMaxScaler().fit(data)

    scaleddata = scaler.transform(data)
    print("scale done!")
    if pca:
        pcaresults = pca_transform(scaleddata, eigmin, export_pca)
        scaleddata = pcaresults['pca_transformed']

    if nrndsample == "all":
        datatotrain = scaleddata
    elif nrndsample < scaleddata.shape[0]:

        random.seed(seed)
        random_indices = random.sample(range(scaleddata.shape[0]), nrndsample)
        datatotrain = scaleddata[random_indices]

    print("kmeans training using a {} x {} matrix".format(datatotrain.shape[0],
                                                          datatotrain.shape[1]))
    kmeansclusters = KMeans(n_clusters=nclusters).fit(datatotrain)
    clusters = kmeansclusters.predict(scaleddata)
    output = {
        'labels': clusters,
        'kmeans_model': kmeansclusters,
        'scale_model': scaler,
        'pca_model': np.nan
    }
    if export_pca:
        output['pca_model'] = pcaresults['pca_model']

    return output


def img_rf_classification(xrdata, model, ml_features):
    idvarsmodel = [list(xrdata.keys()).index(i) for i in ml_features]

    if len(idvarsmodel) == len(ml_features):
        npdata, idsnan = data_processing.from_xarray_to_table(xrdata,
                                                              nodataval=xrdata.attrs['nodata'],
                                                              features_names=ml_features)

        # model prediction
        ml_predicition = model.predict(npdata)

        # organize data as image
        height = xrdata.dims['y']
        width = xrdata.dims['x']

        return data_processing.assign_valuestoimg(ml_predicition,
                                                  height,
                                                  width, idsnan)
