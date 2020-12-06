

''' Takes an image, resizes the image and fills the non existing space
with custom color 
'''
import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur

def resize_sq_fill_image(img, sq_size=224, fill_col=(255,255,255),
                 interpolation=Image.ANTIALIAS):
    
    # can be PIL.Image.Image or np.array
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    elif type(img) == PIL.Image.Image:
        pass
    else:
        print("ERROR: image format unknown, please give np.array or PIL.Image.Image")
    # PIL.Image.Image format from here

    w, h = img.size
    if h != w:
        dif = h if h > w else w
        new_img = Image.new('RGB', (dif, dif), fill_col)
        new_img.paste(img, (int((dif - w) / 2), int((dif - h) / 2)))
        img = new_img
    
    # Change format to np.array and resize with cv2
    new_img = cv2.resize(np.asarray(img), (sq_size, sq_size), interpolation)
    return np.array(new_img)

def equalize_hist(img, clahe_on=False, gridsize=3, clipLimit=2.0):

    if clahe_on:
        clahe = cv2.createCLAHE(clipLimit=clipLimit,
                                tileGridSize=(gridsize,
                                              gridsize))

    if len(img.shape)==3:

        # -> RGB image
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # RGB to LAB
        lab_planes = cv2.split(lab)
        if clahe_on: # contrast limited adaptative histograme equalizer
            lab_planes[0] = clahe.apply(lab_planes[0])
        else: # simple global histogram equalizer
            lab_planes[0] = cv2.equalizeHist(lab_planes[0])
        lab = cv2.merge(lab_planes)
        new_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # convert to bgr

    elif len(img.shape)==2:

        # -> grey scale image
        if clahe_on:
            new_img = clahe.apply(img)
        else:
            new_img = cv2.equalizeHist(img)

    return new_img

import copy

def apply_threshold(img):

    new_img = copy.deepcopy(img)

    if len(img.shape)==3:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)

    ret, thresh = cv2.threshold(new_img, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ## adaptive thershold
    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY,11,2)
    
    return thresh


def gauss_blur(img, radius): 
    return cv2.GaussianBlur(img, (radius, radius),0)

def med_blur(img, radius):
    return cv2.medianBlur(img, radius)

def rgb_to_grey(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def fast_non_loc_means_denois(img):
    return cv2.fastNlMeansDenoising(img)

'''
Takes an image and apply a chosen sequence of transformations
'''

import cv2
import numpy as np
from PIL import Image
import copy

def preproc_image(img, list_preproc_tup): # {'list_preproc_tup': xxxx}

    new_img = copy.deepcopy(img)
    dict_transf = {'resize': resize_sq_fill_image,
                   'gauss_bl': gauss_blur,
                   'med_bl': med_blur,
                   'rgb_to_grey': rgb_to_grey,
                   'nl_denois': fast_non_loc_means_denois,
                   'equalize' : equalize_hist,
                   'thresh': apply_threshold,
                   }

    for name, dict_params in list_preproc_tup:
        new_img = dict_transf[name](new_img, **dict_params)

    return np.array(new_img)

'''
Show one image through specific preproc function and preproc params
'''
import matplotlib.pyplot as plt
import copy

def show_img_and_hist(img_orig=None, img=None, rgb_hist_on=True, figsize=(8,4),
                      preproc_func=None, preproc_params=None):

    # if img_orig -> preproc to apply
    if img_orig is not None:
        fig = plt.figure(figsize=figsize)
        ax1, ax2, ax3, ax4  = fig.add_subplot(221),\
                                fig.add_subplot(222),\
                                fig.add_subplot(223),\
                                fig.add_subplot(224)

        if preproc_func is None:
            print("BEWARE, no preprocessing function was given")
        img = preproc_func(img_orig, **preproc_params)
                            
        ax1.imshow(img_orig, cmap='gray')
        ax3.imshow(img, cmap='gray')
        
        if rgb_hist_on and len(img_orig.shape)==3:
        # Show the 3 histograms (rgb)
            for i, col in enumerate(['r', 'g', 'b']):
                ax2.hist(img_orig[:,:,i].flatten(), bins=range(256),
                            color=col, alpha = 0.5, zorder=10)
        if rgb_hist_on and len(img.shape)==3:
        # Show the 3 histograms (rgb)
            for i, col in enumerate(['r', 'g', 'b']):
                ax4.hist(img[:,:,i].flatten(), bins=range(256),
                            color=col, alpha = 0.5, zorder=10)
        # show global histogram
        ax2.hist(img_orig.flatten(), bins=range(256), color='dimgrey')
        ax4.hist(img.flatten(), bins=range(256), color='dimgrey')

        ax1.set_title('Original image')
        ax2.set_title('Original histogram')
        ax3.set_title(f'Image after preprocessing')
        ax4.set_title(f'Histogram after preprocessing')

    else: # show img only

        if img is not None:
            fig = plt.figure(figsize=figsize)
            ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
            ax1.imshow(img, cmap='Greys_r')
        else:
            print("ERROR: No image given at all !!!")
        
        if rgb_hist_on and len(img.shape)==3:
        # Show the 3 histograms (rgb)
            for i, col in enumerate(['r', 'g', 'b']):
                ax2.hist(img[:,:,i].flatten(), bins=range(256),
                            color=col, alpha = 0.5, zorder=10)
        # show only global histogram
        ax2.hist(img.flatten(), bins=range(256), color='dimgrey')

        ax1.set_title(f'Image')
        ax2.set_title(f'Histogram')

    plt.tight_layout()
    plt.show()

''' can be used to test preprocessing on representative samples
of each category
'''
import matplotlib.pyplot as plt

def print_thumbnails_from_df(ser_pict, li_files, preproc_func=None,
                             preproc_params=None, n_rows=1,
                             figsize=(15,2), fig=None,
                             li_im_title=None, title=None):

    n_tot = len(li_files)
    n_cols = (n_tot//n_rows)+((n_tot%n_rows)>0)*1
    fig = plt.figure(figsize=figsize) if fig is None else fig
    for i, ind in enumerate(li_files,1):
        img = ser_pict.loc[ind]
        if preproc_func is not None:
            img = preproc_func(img, **preproc_params)
        ax = fig.add_subplot(n_rows,n_cols,i)
        if len(img.shape)==3:
            ax.imshow(img)
            if li_im_title is not None:
                ax.set_title(li_im_title[i-1])
        else:
            ax.imshow(img, cmap='Greys_r')
            if li_im_title is not None:
                ax.set_title(li_im_title[i-1])
        ax.set_axis_off()
        if title is not None:
            plt.suptitle(title, fontweight='bold')
    plt.show()

import matplotlib.pyplot as plt

def print_sample_by_from_df(ser_pict, sercat, n_img=10, n_rows=1,
                            preproc_func=None, preproc_params=None,
                            figsize=(20,2)):

    gb = ser_pict.groupby(sercat)
    for name, sub_df in gb:
        li_files = sub_df.sample(n_img).index
        print_thumbnails_from_df(ser_pict, li_files,
                                 preproc_func,
                                 preproc_params,
                                 n_rows=n_rows,
                                 figsize=figsize,
                                 title=name)


'''
Class to get from the big dataframe df_pict (containing all
the preprocessed images) one series of preprocessed images (column : n_col_img)
and unfold the data to get a dataframe with each pixel as a column
'''
from sklearn.base import BaseEstimator, TransformerMixin

class GetImageFromDf(BaseEstimator, TransformerMixin):

    def __init__(self, to_df=True, n_col_img=None):
        self.to_df = to_df
        self.n_col_img = n_col_img

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None): # X is the big df_pict dataframe
        if self.to_df:
            # get the pd.DataFrame with flattened pixels values corresponding to n_col_img
            ser_list = X[self.n_col_img].apply(lambda x: x.flatten())
            X_trans = pd.DataFrame.from_dict(dict(zip(ser_list.index,
                                                ser_list.values))).T
        else:
            # get the pd.Series od np.arrays corresponding to n_col_img
            X_trans = X[self.n_col_img]
        return X_trans

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


'''
Transfomer Class that creates BoVW from a pd.Series of images (as np.arrays)
It computes all the descriptors (SIFT) of each image (np.arrays of 128 vectors)
and one big list of all the descriptors, then applies a k-means clustering
 on the list of all descriptors to get an array of the centroids
 (nb of clusters is the nb of visual words).
Then computes the descriptors of each image (nb of descr. per image tunable),
 gives for each descriptor the label of the nearest visual word
and builds the list of the visual words in each image
Finally returns a dataframe of the Bag of Visual Words.

EXAMPLE :
bovw_extractor = BoVWExtractor(n_features=150,
                               n_vwords=20)
df_bovw = bovw_extractor.fit_transform(df_pict['img_grey'])
'''

from sklearn.cluster import KMeans
from collections import Counter
import cv2
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BoVWExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, n_features=20, n_vwords=50):
        self.n_features = n_features
        self.n_vwords = n_vwords

    def __extract_sift_features(self, ser_images, n_features):
        descriptors = []
        all_descriptors = []
        sift = cv2.SIFT_create(nfeatures=n_features,
                               nOctaveLayers=3, # default val
                               contrastThreshold=0.04,  # default val
                               edgeThreshold=10,  # default val
                               sigma=1.6)  # default val
        ser_descriptors = pd.Series(dtype=int)
        for i, img in ser_images.iteritems():
            kp, desc = sift.detectAndCompute(img, None)
            desc = [np.zeros((128,))] if desc is None else desc # in case no descriptor
            all_descriptors.extend(list(desc))
            descriptors.append(desc)
        ser_descriptors = pd.Series(descriptors,
                                    index=ser_images.index)
        return ser_descriptors, all_descriptors

    def __select_visual_words(self, n_visual_words, descriptors_list):
        clusterer = KMeans(n_clusters = n_visual_words,
                        random_state=14)
        clusterer.fit(pd.DataFrame(descriptors_list))
        visual_words = clusterer.cluster_centers_ 
        return visual_words, clusterer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # extract the features for each image and get the whole list of descriptors
        ser_descriptors, all_descriptors = \
                self.__extract_sift_features(X, # pd.Series of images as np.arrays
                                            self.n_features)
        
        # select the visual words from all the descriptors (clustering)
        visual_words, vw_predictor = self.__select_visual_words(self.n_vwords,
                                                                all_descriptors)

        # loop on the items(pictures) to generate a list of vw labels lists
        vw_list = []
        ind_img_list = []
        for n_img, desc_list in ser_descriptors.iteritems(): 
            vwords = vw_predictor.predict(desc_list)
            vw_list.append(vwords)

        # convert the list of lists in a sorted BoVW dataframe
        df_BovW = pd.DataFrame([Counter(x) for x in vw_list],
                            index = ser_descriptors.index).fillna(0)
        df_BovW = df_BovW[sorted(df_BovW.columns)]

        return df_BovW

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


'''
Transfomer Class that extract features from a pd.Series of images (as np.arrays)
It extract all the features of each image by transfer learning on one chosen
pretrained CNN.

EXAMPLE :
cnn_feat_extractor = CNNFeaturesExtractor(XXXXXXXXX)
df_cnnfeat = cnn_feat_extractor.fit_transform(df_pict['img_grey'])
'''

from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow.keras.applications as app
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB0, EfficientNetB7
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Flatten

class CNNFeaturesExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, n_layers_remv=1,
                 cnn_name='resnet',
                 gap_or_flat='global_max',
                 dropout_rate=0,
                 final_dense_layer=False):

        self.n_layers_remv = n_layers_remv
        self.cnn_name = cnn_name
        self.gap_or_flat = gap_or_flat
        self.dropout_rate = dropout_rate
        self.final_dense_layer = final_dense_layer

    def fit(self, X, y=None):
        return self

    def __get_cnn(self):
        size_sq = 224
        dict_cnn={'resnet': ResNet50(weights='imagenet',
                                     include_top=False, # remove the last layer
                                     input_shape=(size_sq,size_sq,3)),
                  'vgg': VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(size_sq,size_sq,3)),
                  'inceptV3': InceptionV3(weights="imagenet",
                                          include_top=False,
                                          input_shape=(size_sq,size_sq,3),),
                  'effnetB0': EfficientNetB0(weights='imagenet',
                                             include_top=False,
                                             input_shape=(size_sq,size_sq,3)),
                  'effnetB7': EfficientNetB7(weights='imagenet',
                                             include_top=False,
                                             input_shape=(size_sq,size_sq,3))}

        return dict_cnn[self.cnn_name]

    def __get_preproc(self):
        dict_preproc_kw={'resnet': {'mode': 'caffe'},
                         'vgg': {'mode': 'caffe'},
                         'inceptV3': {'mode': 'tf'},
                         'effnetB0': {'mode': 'torch'},
                         'effnetB7': {'mode': 'torch'}}
        return dict_preproc_kw[self.cnn_name]

    def __add_layer(self):

        dict_layer={'glob_max': GlobalMaxPooling2D(name="gap"),
                    'glob_avg': GlobalAveragePooling2D(name="gap"),
                    'flatten': Flatten(name="flatten")}

        return dict_layer[self.gap_or_flat]

    def transform(self, X, y=None):

        # getting the convolutional base of a ResNet50 CNN 
        cnn_conv_base = self.__get_cnn()

        ### Creating the CNN model
        # add cnn base to our model
        cnn_model = Sequential()
        cnn_model.add(cnn_conv_base)
        # add a pooling or flattening layer
        cnn_model.add(self.__add_layer())
        # add a dropout layer
        if self.dropout_rate > 0:
            cnn_model.add(Dropout(self.dropout_rate, name="dropout_out"))
        # add a final dense layer
        if self.final_dense_layer:
            # model.add(Dense(256, activation='relu', name="fc1"))
            cnn_model.add(Dense(2, activation="softmax", name="fc_out"))

        # loop on the items(pictures) to preprocess images and extract features
        features = []
        for img in X: # X: pd.Series of np.arrays (img)
            # reshape img (np.array) to get a tensor
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            # preprocess images using convenient preprocessing
            img = imagenet_utils.preprocess_input(img, data_format=None,
                                                  mode=self.__get_preproc())
            # get the extracted features
            features.append(cnn_model.predict(img).reshape(-1))
        
        # for cnn features, get a dataframe with features for each item
        df_trans = pd.DataFrame.from_dict(dict(zip(df_pict.index,
                                                   features))).T

        return df_trans


    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

