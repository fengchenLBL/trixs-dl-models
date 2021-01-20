#!/usr/bin/env python
# coding: utf-8

# # Main Notebook for Training Models and Generating Figures

# ### Here are some flags which will affect the way the notebook executes and what data is written.

# In[ ]:


import sys
#!{sys.executable} -m pip install pymatgen
#!{sys.executable} -m pip install trixs
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
#import eli5 # not needed
#from eli5.sklearn import PermutationImportance # not needed
from keras.regularizers import l2 # Regularization in Keras L2: Sum of the squared weights.
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import zeros, newaxis
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D, AveragePooling1D
from keras.utils import to_categorical


# In[ ]:


import sys
#!{sys.executable} -m pip install numpy --upgrade
#!{sys.executable} -m pip install swish-activation
#!{sys.executable}  -m pip install git+https://github.com/keras-team/keras-tuner.git
#!{sys.executable} -m pip install autokeras

#!{sys.executable} -m pip install pymatgen
#!{sys.executable} -m pip install trixs

import tensorflow as tf
import autokeras as ak
import pandas as pd
import numpy as np


# In[ ]:


import sys
import os
import sklearn
import json
import numpy as np
from collections import Counter
from scipy.stats import norm
from typing import List


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
#get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm, tqdm_notebook
from pprint import pprint

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator

#from trixs.machine_learning.benchmarks import precision_recall_matrix, confusion_dict

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

from trixs.spectra.spectrum_featurize import polynomialize_by_idx, gauge_polynomial_error


# In[ ]:


# Runs notebook in a mode which uses max-normalized spectra 
#(figures for this can be found in the paper's supplemental information.)
# Maintext figures set this variable to FALSE.
use_max_normalized = False
norm_str = 'max' if use_max_normalized else 'feff_dl_moreData_P4'

# Flag for using validation data (for model characterization)
# or testing data (should only be done once the previous process is complete).
# Default to testing data.
use_test = True

# Set random seed to be used as argument for other functions.
rseed = 42


# In[ ]:


storage_directory = './spectral_data'

figure_write_folder = "./figures_feffnorm" if not use_max_normalized else './figures_maxnorm'
try: 
    os.mkdir(figure_write_folder) 
except OSError as error: 
    pass
np.random.seed(rseed)


# In[ ]:


print("The publication uses SKlearn version 0.21.3. Yours:",sklearn.__version__)


# ## Define domains which will be used for x-axis labels later, as well as define the elements which will be imported for use

# In[ ]:


target_elements_groups=[('Ti','O'),('V','O'),('Cr','O'),
                        ('Mn','O'),('Fe','O'),('Co','O'),
                        ('Ni','O'),('Cu','O')]

x_domains = {  ('Co','O'):  np.linspace(7713.5, 7765.83,100),
               ('Fe','O'): np.linspace(7115.0, 7167.764,100),
               ('V','O'):  np.linspace(5468.0, 5520.631,100),
               ('Cu','O'): np.linspace( 8987.5, 9039.712,100),
               ('Ni','O'): np.linspace( 8336.5 ,8388.723,100),
               ('Cr','O'): np.linspace(5993.1, 6045.686,100),
               ('Mn','O'): np.linspace(6541.7, 6594.417,100),
               ('Ti','O'): np.linspace(4969.0, 5021.024,100)}

colors_by_pair = {('Ti','O'):'orangered',
                  ('V','O'):'darkorange',
                  ('Cr','O'):'gold',
                  ('Mn','O'):'seagreen',
                  ('Fe','O'):'dodgerblue',
                  ('Co','O'):'navy',
                  ('Ni','O'):'rebeccapurple',
                  ('Cu','O'):"mediumvioletred"}

pair_to_name={'Ti':"Titanium",'V':'Vanadium',
              'Cr':'Chromium','Mn':"Manganese",
              'Fe':"Iron",'Co':"Cobalt",
             'Ni':'Nickel','Cu':'Copper'}


# # Load in Pointwise Data

# # Set up Precision / Recall Matrix

# In[ ]:


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def precision_recall(fits: List, labels: List, target)->List[float]:
    """
    Computes the precision and recall and F1 score
    for an individual class label 'target',
    which can be any object with an equivalence relation via ==
    :param fits:
    :param labels:
    :param target:
    :return:
    """
    N = len(labels)

    # Generate the counts of true and false positives
    true_positives = len([True for i in range(N)
                          if (fits[i] == target and labels[i] == target)])
    false_positives = len([True for i in range(N)
                           if (fits[i] == target and labels[i] != target)])
    false_negatives = len([True for i in range(N)
                           if (fits[i] != target and labels[i] == target)])

    if true_positives == 0:
        return [0, 0, 0]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2.0 * precision * recall / (precision + recall)
    return [precision, recall, f1]


def precision_recall_matrix(fits: List, labels: List, classes: List):
    """
    Computes the precision and recall and F1 score for a set of classes at once

    :param fits:
    :param classes:
    :param labels:
    :return:
    """
    results = []
    for cls in classes:
        results.append(precision_recall(fits, labels, cls))
    return np.array(results)

def avg_f1_score(guesses,labels):
    f1_score = precision_recall_matrix(guesses,labels,[4,5,6])
    return np.mean([np.round(100*x[2],1) for x in f1_score])


# # Load in Train/Test sets

# In[ ]:



# TT stands for Train-Test
# c = coord, b = bader, md = mean distance
ttc_by_pair = {pair:{} for pair in target_elements_groups}
ttb_by_pair = {pair:{} for pair in target_elements_groups}
ttmd_by_pair = {pair:{} for pair in target_elements_groups}

for pair in target_elements_groups:
    for key in ['train_x','train_y','valid_x','valid_y','test_x','test_y']:
        ttc_by_pair[pair][key] =np.load(f'./model_data/{pair[0]}_coord_{key}.npy')
        ttb_by_pair[pair][key] =np.load(f'./model_data/{pair[0]}_bader_{key}.npy')
        ttmd_by_pair[pair][key] =np.load(f'./model_data/{pair[0]}_md_{key}.npy')
        
# Quickly normalize the input X spectra if toggled at top of notebook.
if use_max_normalized:
    for pair in target_elements_groups:
        for key in ['train_x','valid_x','test_x']:
            ttc_by_pair[pair][key] = np.array([array / np.max(array) for array in ttc_by_pair[pair][key][:]])
            ttb_by_pair[pair][key] = np.array([array / np.max(array) for array in ttb_by_pair[pair][key][:]])
            ttmd_by_pair[pair][key] = np.array([array / np.max(array) for array in ttmd_by_pair[pair][key][:]])
            


# # Main Cell 1:
# # Random Forests Trained using Pointwise spectra

# In[ ]:


# Flag to run or not run the cell
run = True
# Flag to display plots inline
show_plots = False
print("Commencing run...")

accuracies = {}
deviations = {}
all_data_values = []

accuracies_nn = {}
deviations_nn = {}
all_data_values_nn = []

md_perf_by_pair={}
bader_perf_by_pair={}
models_by_pair = {}
means_by_pair = {}

md_perf_by_pair_nn={}
bader_perf_by_pair_nn={}
models_by_pair_nn = {}
means_by_pair_nn = {}


use_test = True

# Flags for your own experimentaion purposes if you'd like to focus on one task or another.
# N O T E ! The rest of the notebook assumes all three flags are on!
run_coord = True
run_bader = True
run_md = True

# How many times to repeat the training of the random forests 
# with different random seeds used
# for the training, to generate error bars 
# on the feature RANKing (hence, RANK REPEAT).

# The publication uses 10 total trainings, so RANK_REPEAT is set here to 9.
RANK_REPEAT = 9


# The hyperparameter N_ESTIMATORS is controllable here because running the notebook with
# a smaller number of estimators may be desirable to verify that things are working.
# Set to 300 by default-- as was used in the publication.


# ## Define nerual network models

# In[ ]:


# define baseline model for coord w/o regularization term L2
def baseline_model_c():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=100, activation='relu')) # the input X dim is 100
    model.add(Dense(100, activation='relu')) # more hidden layer
    model.add(Dense(50, activation='relu')) # more hidden layer
    model.add(Dense(3, activation='softmax')) # the outpu Y dim is 3
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define regression base_model for b w/o regularization term L2
def base_model_b():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, activation='relu')) # more hidden layer
    model.add(Dense(50, activation='relu')) # more hidden layer
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# define regression base_model for md w/o regularization term L2
def base_model_md():
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu')) # more hidden layer
    model.add(Dense(10, kernel_initializer='normal', activation='relu')) # more hidden layer
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# ## Define CNN models

# In[ ]:


def yconvert(y):
    if y == 4:
        return([1,0,0])
    if y == 5:
        return([0,1,0])
    if y == 6:
        return([0,0,1])

def ycnn(yc_train):
    return np.array([yconvert(y) for y in yc_train])

def xcnn(xc_train):
    return(xc_train[:, :, newaxis])


# cnn model for corrdinate
def cnn_model_c():
    n_features, n_timesteps, n_outputs = 100, 1, 3
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(n_features, n_timesteps)))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#nn_c = KerasClassifier(build_fn=cnn_model_c, epochs=10, batch_size=32, verbose=1)
#nn_c.fit(xcnn(xc_train), ycnn(yc_train))
#print('training accuracy: ', nn_c.score(xcnn(xc_train), ycnn(yc_train)))
#print('testing accuracy: ', nn_c.score(xcnn(xc_valid), ycnn(yc_valid)))


# cnn model for bader
def cnn_model_b():
    n_features, n_timesteps, n_outputs = 100, 1, 1
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, kernel_initializer='normal', activation='relu', input_shape=(n_features, n_timesteps)))
    model.add(Conv1D(filters=64, kernel_size=5, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#nn_b = KerasRegressor(build_fn=cnn_model_b, epochs=100, batch_size=32, verbose=1)
#nn_b.fit(xcnn(xb_train), yb_train)
#print("R2:", r2_score(yb_valid, nn_b.predict(xcnn(xb_valid)))) # R-Squared
#print("MSE:", np.mean(np.abs(nn_b.predict(xcnn(xb_valid)) - yb_valid)))

# cnn model for md
def cnn_model_md():
    n_features, n_timesteps, n_outputs = 100, 1, 1
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, kernel_initializer='normal', activation='relu', input_shape=(n_features, n_timesteps)))
    model.add(Conv1D(filters=64, kernel_size=5, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#nn_md = KerasRegressor(build_fn=cnn_model_md, epochs=10, batch_size=N_EPOCHS, verbose=1)
#nn_md.fit(xcnn(xmd_train), ymd_train)
#print("R2:", r2_score(ymd_valid, nn_md.predict(xcnn(xmd_valid)))) # R-Squared
#print("MSE:", np.mean(np.abs(nn_md.predict(xcnn(xmd_valid)) - ymd_valid)))


# ## Data Augmentation

# In[ ]:


## Data Augmentation: x +/- 0.5;
def x_augmentation(x, delta=0.5):
    x0 = list(x)
    # add noise on y-axis
    x1 = [np.array(x[k]) + np.array(delta*np.random.normal(0,1,100)) for k in range(len(x))]
    x2 = [np.array(x[k]) - np.array(delta*np.random.normal(0,1,100)) for k in range(len(x))]
    # add noise on x-axis
    x1 = [np.array(list(tmp[1:]) + list([tmp[-1]])) if np.random.normal(0)>0 else np.array(list([tmp[0]]) + list(tmp[:-1])) for tmp in x1]
    x2 = [np.array(list(tmp[1:]) + list([tmp[-1]])) if np.random.normal(0)>0 else np.array(list([tmp[0]]) + list(tmp[:-1])) for tmp in x2]
    x0.extend(list(x1))
    x0.extend(list(x2))
    return(np.array(x0))

## Data Augmentation: randomly add any value between -0.03 and +0.03 
def y_augmentation(y, delta=0.03):
    y0 = list(y)
    noise1 = np.random.normal(0,1,len(y)) 
    noise2 = np.random.normal(0,1,len(y)) 
    #y1 = y + y*delta*noise1
    #y2 = y + y*delta*noise2
    y1 = y + delta*noise1
    y2 = y + delta*noise2
    y0.extend(y1)
    y0.extend(y2)
    return(np.array(y0))


# In[ ]:


from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count() 
print("Number of Cores: ", num_cores)


# In[ ]:


#N_ESTIMATORS = 300
#N_EPOCHS = 150
# RANK_REPEAT = 9

raw_data = {}

N_ESTIMATORS = 300
N_EPOCHS = 300
RANK_REPEAT = 1
VERBOSE = 0
pair = target_elements_groups[0]

pair = target_elements_groups[0]
for pair in tqdm(target_elements_groups, ncols=90, desc='target_elements_groups'):
    if not run:
        continue
    
    # Instantiate each random forest model
    raw_data[pair] = {}
    forest_c = RandomForestClassifier(random_state=rseed,
                                      n_estimators=N_ESTIMATORS,
                                      max_depth =35, 
                                      max_features = 8, min_samples_leaf = 1,
                                      min_samples_split = 2,
                                      class_weight=None,
                                      n_jobs=4)


    forest_b = RandomForestRegressor(random_state=rseed,
                                     criterion='mse',max_depth=35,
                                     max_features=8, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=N_ESTIMATORS,
                                     n_jobs=4)
    
    forest_md = RandomForestRegressor(random_state=rseed,
                                      criterion='mse',max_depth=35,
                                      max_features=8, max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_impurity_split=None,
                                      min_samples_leaf=1, min_samples_split=2,
                                      min_weight_fraction_leaf=0.0, n_estimators=N_ESTIMATORS,
                                      n_jobs=4)

    # Instantiate each neural nets model
    nn_c = KerasClassifier(build_fn=baseline_model_c, epochs=N_EPOCHS, batch_size=32, verbose=VERBOSE)
    
    nn_b = Pipeline([('standardize', StandardScaler()),
                     ('estimator', KerasRegressor(build_fn=base_model_b, epochs=N_EPOCHS, batch_size=32, verbose=VERBOSE))])
    nn_md = Pipeline([('standardize', StandardScaler()),
                      ('estimator', KerasRegressor(build_fn=base_model_md, epochs=N_EPOCHS, batch_size=32, verbose=VERBOSE))])

    #nn_c = KerasClassifier(build_fn=cnn_model_c, epochs=N_EPOCHS, batch_size=32, verbose=VERBOSE)
    #nn_b = KerasRegressor(build_fn=cnn_model_b, epochs=N_EPOCHS, batch_size=32, verbose=VERBOSE)
    #nn_md = KerasRegressor(build_fn=cnn_model_md, epochs=N_EPOCHS, batch_size=32, verbose=VERBOSE)

    #############################################
    # COORDINATION
    #############################################
    
    xc_train = ttc_by_pair[pair]['train_x'] 
    yc_train = ttc_by_pair[pair]['train_y']
    #xc_train = x_augmentation(xc_train, delta=0.03)
    #yc_train = y_augmentation(yc_train, delta=0.0)
    xc_valid = ttc_by_pair[pair]['valid_x'] 
    yc_valid = ttc_by_pair[pair]['valid_y']
    #xc_valid = x_augmentation(xc_valid, delta=0.03)
    #yc_valid = y_augmentation(yc_valid, delta=0.0)    

    Y = list(yc_train) + list(yc_valid)
    X = list(xc_train) + list(xc_valid)
    #xc_train, xc_valid, yc_train, yc_valid = train_test_split(np.array(X), np.array(Y), test_size=0.33, random_state=42)
            
    if run_coord:
        def processInput_c(i, forest_c=forest_c, nn_c=nn_c):
            #xc_train, xc_valid, yc_train, yc_valid = train_test_split(np.array(X), np.array(Y), test_size=0.33, random_state=i+1)
            forest_c.random_state = rseed+i+1
            forest_c.fit(xc_train,yc_train)
            
            nn_c = ak.StructuredDataClassifier(overwrite=True, max_trials=5)
            nn_c.fit(xc_train, yc_train, epochs=300, verbose=False)
            #nn_perm_c = permutation_importance(nn_c, xc_train, yc_train, n_repeats=2, random_state=1)

            cur_model_f1s = [x[2]*100 for x in  precision_recall_matrix(forest_c.predict(xc_valid),yc_valid,[4,5,6])]
            cur_model_accuracies = forest_c.score(xc_valid,yc_valid)
            cur_model_importances = forest_c.feature_importances_
            
            nn_model_f1s = [x[2]*100 for x in  precision_recall_matrix([int(p) for p in sum(clf.predict(xc_valid).tolist(), [])], yc_valid, [4,5,6])]
            nn_model_accuracies = nn_c.evaluate(xc_valid, yc_valid)[1]
            #nn_model_importances = nn_perm_c.importances_mean/np.sum(nn_perm_c.importances_mean)
            
            return {'cur_model_f1s':cur_model_f1s, 
                    'cur_model_accuracies': cur_model_accuracies, 
                    'cur_model_importances': cur_model_importances, 
                    'nn_model_f1s': nn_model_f1s,
                    'nn_model_accuracies': nn_model_accuracies}
        
        results_c = Parallel(n_jobs=num_cores if num_cores<RANK_REPEAT else RANK_REPEAT)(delayed(processInput_c)(i) for i in tqdm(range(RANK_REPEAT), ncols=90, desc='Cross Validation coord'))
        
        cur_model_importances = [r['cur_model_importances'] for r in results_c]
        #nn_model_importances = [r['nn_model_importances'] for r in results_c]
        cur_model_accuracies = [r['cur_model_accuracies'] for r in results_c]
        cur_model_f1s = [r['cur_model_f1s'] for r in results_c]
        nn_model_accuracies = [r['nn_model_accuracies'] for r in results_c]
        nn_model_f1s = [r['nn_model_f1s'] for r in results_c]
        #forest_c = results_c[0]['forest_c']
        #nn_c = results_c[0]['nn_c']
        
        importances_mean = np.mean(cur_model_importances,axis=0)
        #nn_importances_mean = np.mean(nn_model_importances,axis=0)
        coord_accuracies_mean = np.mean(cur_model_accuracies)
        coord_f1s_mean = np.mean(cur_model_f1s, axis=0)

        nn_coord_accuracies_mean = np.mean(nn_model_accuracies)
        nn_coord_f1s_mean = np.mean(nn_model_f1s, axis=0)

        if RANK_REPEAT:
            importances_std = np.std(cur_model_importances,axis=0)
            #nn_importances_std = np.std(nn_model_importances,axis=0)
            coord_accuracies_std = np.std(cur_model_accuracies)
            coord_f1s_std = np.std(cur_model_f1s,   axis=0)
            nn_coord_accuracies_std = np.std(nn_model_accuracies)
            nn_coord_f1s_std = np.std(nn_model_f1s,   axis=0)
        else:
            importances_std = np.zeros(len(cur_model_importances))
            coord_accuracies_std = np.zeros(len(cur_model_accuracies))
            coord_f1s_std = np.zeros(len(cur_model_f1s))
            nn_coord_accuracies_std = np.zeros(len(nn_model_accuracies))
            nn_coord_f1s_std = np.zeros(len(nn_model_f1s))


        means_by_pair[str(pair)+'-coord'] = importances_mean

        plt.clf()
        plt.figure(figsize=(16,9))
        plt.errorbar(x_domains[pair],importances_mean,yerr= importances_std,
                     label='RF: $\mu$ $\pm$ 1 $\sigma$ (N={})'.format(1+RANK_REPEAT),
                     color='black',ecolor='blue')
        plt.title("Model Importance Spread \n{} Coordination (Pointwise)".format(pair[0]))
        plt.legend()
        plt.savefig(figure_write_folder+'/{}_{}_all_coord_mean_importances.pdf'.format(pair[0],norm_str),
                    format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        # Store the last model trained for later 
        # use in comparing against polynomial models.
        #models_by_pair[str(pair)+'-Coord'] = forest_c
        #models_by_pair_nn[str(pair)+'-Coord'] = nn_c

        print("Done with Coordination for ",pair)
        
        class_makeup = Counter(yc_valid)
        mode_guess_score = max(class_makeup.values())/sum(class_makeup.values())
        raw_data[pair]['c'] = results_c
    #############################################
    # BADER
    #############################################
    xb_train =  ttb_by_pair[pair]['train_x']
    yb_train =  ttb_by_pair[pair]['train_y']
    #xb_train = x_augmentation(xb_train, delta=0.03)
    #yb_train = y_augmentation(yb_train, delta=0.0)

    xb_valid = ttb_by_pair[pair]['valid_x']
    yb_valid = ttb_by_pair[pair]['valid_y']
    #xb_valid = x_augmentation(xb_valid, delta=0.03)
    #yb_valid = y_augmentation(yb_valid, delta=0.0)

    Y = list(yb_train) + list(yb_valid)
    X = list(xb_train) + list(xb_valid)
    #xb_train, xb_valid, yb_train, yb_valid = train_test_split(np.array(X), np.array(Y), test_size=0.33, random_state=42)
    
    if run_bader:
        def processInput_b(i, forest_b=forest_b, nn_b=nn_b):
            #xb_train, xb_valid, yb_train, yb_valid = train_test_split(np.array(X), np.array(Y), test_size=0.33, random_state=i+1)
            forest_b.random_state = rseed+i+1
            forest_b.fit(xb_train,yb_train)    
            
            ## autoKeras structured data regression
            nn_b = ak.StructuredDataRegressor(overwrite=True, max_trials=5)
            
            nn_b.fit(xb_train, yb_train)
            #nn_perm_b = permutation_importance(nn_b, xb_train, yb_train, n_repeats=2, random_state=1)

            cur_model_importances = forest_b.feature_importances_ 
            cur_model_accuracies = r2_score(yb_valid,nn_b.predict(xb_valid)) # R-Squared
            cur_model_maes = np.mean(np.abs(forest_b.predict(xb_valid) - yb_valid))
            
            nn_model_accuracies = r2_score(yb_valid,nn_b.predict(xb_valid)) # R-Squared
            nn_model_maes = np.mean(np.abs(nn_b.predict(xb_valid) - yb_valid))
            #nn_model_importances = nn_perm_b.importances_mean/np.sum(nn_perm_b.importances_mean)
            
            return {'cur_model_accuracies': cur_model_accuracies, 
                    'cur_model_importances': cur_model_importances, 
                    'cur_model_maes': cur_model_maes,
                    'nn_model_maes': nn_model_maes,
                    'nn_model_accuracies': nn_model_accuracies,
                    'forest_b_prediction': forest_b.predict(xb_valid),
                    'nn_b_prediction': nn_b.predict(xb_valid)}
            
        results_b = Parallel(n_jobs=num_cores if num_cores<RANK_REPEAT else RANK_REPEAT)(delayed(processInput_b)(i) for i in tqdm(range(RANK_REPEAT), ncols=90, desc='Cross Validation BADER'))
        
        cur_model_importances = [r['cur_model_importances'] for r in results_b]
        nn_model_importances = [r['nn_model_importances'] for r in results_b]
        cur_model_accuracies = [r['cur_model_accuracies'] for r in results_b]
        cur_model_maes = [r['cur_model_maes'] for r in results_b]
        nn_model_accuracies = [r['nn_model_accuracies'] for r in results_b]
        nn_model_maes = [r['nn_model_maes'] for r in results_b]
        forest_b_prediction = results_b[0]['forest_b_prediction']
        nn_b_prediction = results_b[0]['nn_b_prediction']
        
        importances_mean = np.mean(cur_model_importances,axis=0)
        #nn_importances_mean = np.mean(nn_model_importances,axis=0)
        bader_accuracies_mean  = np.mean(cur_model_accuracies)
        bader_maes_mean = np.mean(cur_model_maes)

        nn_bader_accuracies_mean  = np.mean(nn_model_accuracies)
        nn_bader_maes_mean = np.mean(nn_model_maes)

        if RANK_REPEAT:
            importances_std  = np.std(cur_model_importances,axis=0)
            #nn_importances_std = np.std(nn_model_importances,axis=0)
            bader_accuracies_std   = np.std(cur_model_accuracies)
            bader_maes_std = np.std(cur_model_maes)
            nn_bader_accuracies_std   = np.std(nn_model_accuracies)
            nn_bader_maes_std = np.std(nn_model_maes)
        else:
            importances_std = np.zeros(len(cur_model_importances))
            bader_accuracies_std = np.zeros(len(cur_model_accuracies))
            bader_maes_std = np.zeros(len(cur_model_maes))
            nn_bader_accuracies_std = np.zeros(len(nn_model_accuracies))
            nn_bader_maes_std = np.zeros(len(nn_model_maes))


        accuracies[str(pair)+'-Bader'] = bader_accuracies_mean
        deviations[str(pair)+'-Bader'] = bader_accuracies_std
        means_by_pair[str(pair)+'-Bader'] = importances_mean

        accuracies_nn[str(pair)+'-Bader'] = nn_bader_accuracies_mean
        deviations_nn[str(pair)+'-Bader'] = nn_bader_accuracies_std

        plt.clf()
        plt.figure(figsize=(16,9))
        plt.errorbar(x_domains[pair],importances_mean, yerr=importances_std,
                     label='RF: $\mu$ $\pm$ 1 $\sigma$ (N={})'.format(1+RANK_REPEAT),
                     color='black',ecolor='red')
        plt.title("Model Importance Spread \nFor {} Bader Charge (Pointwise)".format(pair[0]))
        plt.legend()
        plt.savefig(figure_write_folder+'/{}_{}_all_bader_mean_importances.pdf'.format(pair[0],norm_str),format='pdf',dpi=300,transparent=True,bbox_inches='tight')

        plt.show()

        # Store the last model trained for later 
        # use in comparing against polynomial models.
        models_by_pair[str(pair)+'-Bader'] = forest_b
        models_by_pair_nn[str(pair)+'-Bader'] = nn_b

        #bader_perf_by_pair[pair[0]+'-guesses'] = forest_b.predict(xb_valid)
        bader_perf_by_pair[pair[0]+'-guesses'] = forest_b_prediction
        bader_perf_by_pair[pair[0]+'-labels'] = yb_valid

        #bader_perf_by_pair_nn[pair[0]+'-guesses'] = nn_b.predict(xb_valid)
        bader_perf_by_pair_nn[pair[0]+'-guesses'] = nn_b_prediction
        bader_perf_by_pair_nn[pair[0]+'-labels'] = yb_valid

        print("Done with Bader for ",pair)
        raw_data[pair]['b'] = results_b

    ##############################
    #   MD PART
    ##############################

    xmd_train = np.array(ttmd_by_pair[pair]['train_x'])
    ymd_train = np.array(ttmd_by_pair[pair]['train_y'])
    #xmd_train = x_augmentation(xmd_train, delta=0.03)
    #ymd_train = y_augmentation(ymd_train, delta=0.0)

    xmd_valid = np.array(ttmd_by_pair[pair]['valid_x'])
    ymd_valid = np.array(ttmd_by_pair[pair]['valid_y'])
    #xmd_valid = x_augmentation(xmd_valid, delta=0.03)
    #ymd_valid = y_augmentation(ymd_valid, delta=0.0)
    
    Y = list(ymd_train) + list(ymd_valid)
    X = list(xmd_train) + list(xmd_valid)
    #xmd_train, xmd_valid, ymd_train, ymd_valid = train_test_split(np.array(X), np.array(Y), test_size=0.33, random_state=42)
    
    md_perf_by_pair[pair[0]+'-labels'] = ymd_valid
    
    if run_md:
        def processInput_md(i, forest_md=forest_md, nn_md=nn_md):
            #xmd_train, xmd_valid, ymd_train, ymd_valid = train_test_split(np.array(X), np.array(Y), test_size=0.33, random_state=i+1)
            forest_md.random_state = rseed+i+1
            forest_md.fit(xmd_train,ymd_train)
                        
            ## autoKeras structured data regression
            nn_md = ak.StructuredDataRegressor(overwrite=True, max_trials=5)
            nn_md.fit(xmd_train, ymd_train, epochs=300, verbose=False)
            
            np.mean(np.abs(nn_md.predict(xmd_valid) - ymd_valid))
            nn_md.fit(xmd_train, ymd_train)
            #nn_perm_md = permutation_importance(nn_md, xmd_train, ymd_train, n_repeats=2, random_state=1)

            cur_model_importances = forest_md.feature_importances_
            cur_model_accuracies = r2_score(ymd_valid,forest_md.predict(xmd_valid)) # R-squared
            cur_model_maes = np.mean(np.abs(forest_md.predict(xmd_valid) - ymd_valid))

            nn_model_accuracies = r2_score(ymd_valid,nn_md.predict(xmd_valid)) # R-squared
            nn_model_maes = np.mean(np.abs(nn_md.predict(xmd_valid) - ymd_valid))
            #nn_model_importances = nn_perm_md.importances_mean/np.sum(nn_perm_md.importances_mean)
            
            return {'cur_model_accuracies': cur_model_accuracies, 
                    'cur_model_importances': cur_model_importances, 
                    'cur_model_maes': cur_model_maes,
                    'nn_model_maes': nn_model_maes,
                    'nn_model_accuracies': nn_model_accuracies,
                    'forest_md_prediction': forest_md.predict(xmd_valid),
                    'nn_md_prediction': nn_md.predict(xmd_valid)}

        results_md = Parallel(n_jobs=num_cores if num_cores<RANK_REPEAT else RANK_REPEAT)(delayed(processInput_md)(i) for i in tqdm(range(RANK_REPEAT), ncols=90, desc='Cross Validation MD'))
        
        cur_model_importances = [r['cur_model_importances'] for r in results_md]
        nn_model_importances = [r['nn_model_importances'] for r in results_md]
        cur_model_accuracies = [r['cur_model_accuracies'] for r in results_md]
        cur_model_maes = [r['cur_model_maes'] for r in results_md]
        nn_model_accuracies = [r['nn_model_accuracies'] for r in results_md]
        nn_model_maes = [r['nn_model_maes'] for r in results_md]
        forest_md_prediction = results_md[0]['forest_md_prediction']
        nn_md_prediction = results_md[0]['nn_md_prediction']
        
        importances_mean = np.mean(cur_model_importances,axis=0)
        #nn_importances_mean = np.mean(nn_model_importances,axis=0)
        md_accuracies_mean  = np.mean(cur_model_accuracies,axis=0)
        md_maes_mean = np.mean(cur_model_maes,axis=0)

        nn_md_accuracies_mean  = np.mean(nn_model_accuracies)
        nn_md_maes_mean = np.mean(nn_model_maes)

        if RANK_REPEAT:
            importances_std  = np.std(cur_model_importances,axis=0)
            #nn_importances_std = np.std(nn_model_importances,axis=0)
            md_accuracies_std   = np.std(cur_model_accuracies,axis=0)
            md_maes_std = np.std(cur_model_maes,axis=0)
            nn_md_accuracies_std   = np.std(nn_model_accuracies)
            nn_md_maes_std = np.std(nn_model_maes)
        else:
            importances_std  = np.zeros(len(cur_model_importances),axis=0)
            md_accuracies_std   = np.zeros(len(cur_model_accuracies),axis=0)
            md_maes_std = np.zeros(len(cur_model_maes),axis=0)
            nn_md_accuracies_std = np.zeros(len(nn_model_accuracies))
            nn_md_maes_std = np.zeros(len(nn_model_maes))

        means_by_pair[str(pair)+'-md'] = importances_mean

        plt.clf()
        plt.figure(figsize=(16,9))
        plt.errorbar(x_domains[pair],importances_mean,yerr= importances_std,
                     label='RF: $\mu$ $\pm$ 1 $\sigma$ (N={})'.format(1+RANK_REPEAT),
                     color='black',ecolor='green')
        plt.title("Model Importance Spread \nFor {} Mean Distance (Pointwise)".format(pair[0]))
        plt.legend()
        plt.savefig(figure_write_folder+'/{}_{}_all_md_mean_importances.pdf'.format(pair[0],norm_str),format='pdf',dpi=300,transparent=True,bbox_inches='tight')
        plt.show()

        models_by_pair[str(pair)+'-Mean'] = forest_md
        models_by_pair_nn[str(pair)+'-Mean'] = nn_md

        # Used in constructing parity plots later

        #md_perf_by_pair[pair[0]+'-guesses'] = forest_md.predict(xmd_valid) 
        md_perf_by_pair[pair[0]+'-guesses'] = forest_md_prediction
        #md_perf_by_pair_nn[pair[0]+'-guesses'] = nn_md.predict(xmd_valid)  
        md_perf_by_pair_nn[pair[0]+'-guesses'] = nn_md_prediction
    
        print("Done with mean distance")
        raw_data[pair]['md'] = results_md
        
    if not (run_bader and run_md and run_coord):
        continue
    
       
    if RANK_REPEAT:
        #AVERAGES
        accuracies[str(pair)+'-Coord'] = np.round(coord_accuracies_mean*100,4)
        accuracies[str(pair)+'-Coord-F1'] = np.round(coord_f1s_mean,2)
        accuracies[str(pair)+'-GuessMode'] = np.round(mode_guess_score*100,2)
        
        accuracies[str(pair)+'-Bader'] = np.round(bader_accuracies_mean*100,4)
        accuracies[str(pair)+'-Bader-MAE'] = np.round(bader_maes_mean,2)
        
        accuracies[str(pair)+'-MeanDist'] = np.round(md_accuracies_mean*100,4)
        accuracies[str(pair)+'-MeanDist-MAE'] = np.round(md_maes_mean,3)


        accuracies_nn[str(pair)+'-Coord'] = np.round(nn_coord_accuracies_mean*100,4)
        accuracies_nn[str(pair)+'-Coord-F1'] = np.round(nn_coord_f1s_mean,2)
        accuracies_nn[str(pair)+'-GuessMode'] = np.round(mode_guess_score*100,2)
        
        accuracies_nn[str(pair)+'-Bader'] = np.round(nn_bader_accuracies_mean*100,4)
        accuracies_nn[str(pair)+'-Bader-MAE'] = np.round(nn_bader_maes_mean,2)
        
        accuracies_nn[str(pair)+'-MeanDist'] = np.round(nn_md_accuracies_mean*100,4)
        accuracies_nn[str(pair)+'-MeanDist-MAE'] = np.round(nn_md_maes_mean,3)
        
        # DEVIATIONS
        deviations[str(pair)+'-Coord'] = np.round(coord_accuracies_std*100,4)
        deviations[str(pair)+'-Coord-F1'] = np.round(coord_f1s_std,4)
        
        deviations[str(pair)+'-Bader'] = np.round(bader_accuracies_std*100,4)
        deviations[str(pair)+'-Bader-MAE'] = np.round(bader_maes_std,4)
        
        deviations[str(pair)+'-MeanDist'] = np.round(md_accuracies_std*100,4)
        deviations[str(pair)+'-MeanDist-MAE'] = np.round(md_maes_std,4)                    

        deviations_nn[str(pair)+'-Coord'] = np.round(nn_coord_accuracies_std*100,4)
        deviations_nn[str(pair)+'-Coord-F1'] = np.round(nn_coord_f1s_std,4)
        
        deviations_nn[str(pair)+'-Bader'] = np.round(nn_bader_accuracies_std*100,4)
        deviations_nn[str(pair)+'-Bader-MAE'] = np.round(nn_bader_maes_std,4)
        
        deviations_nn[str(pair)+'-MeanDist'] = np.round(nn_md_accuracies_std*100,4)
        deviations_nn[str(pair)+'-MeanDist-MAE'] = np.round(nn_md_maes_std,4)                    
    else:
        accuracies[str(pair)+'-Coord'] = np.round(forest_c.score(xc_valid,yc_valid),2)
        accuracies[str(pair)+'-Coord-F1'] = np.round(avg_f1_score(guesses=guesses,labels=yc_valid),2)
        accuracies[str(pair)+'-GuessMode'] = np.round(mode_guess_score,2)
        
        accuracies[str(pair)+'-Bader'] = np.round(forest_b.score(xb_valid,yb_valid),2)
        accuracies[str(pair)+'-Bader-MAE'] = np.round(np.abs(forest_b.predict(xb_valid)-yb_valid).mean(),2)
        
        accuracies[str(pair)+'-MeanDist'] = np.round(forest_md.score(xmd_valid,ymd_valid),2)
        accuracies[str(pair)+'-MeanDist-MAE'] = np.round(np.abs(forest_md.predict(xmd_valid)-ymd_valid).mean(),2)
                         
        deviations[str(pair)+'-Coord'] = 0
        deviations[str(pair)+'-Coord-F1'] = 0
        deviations[str(pair)+'-GuessMode'] =0
        
        deviations[str(pair)+'-Bader'] = 0
        deviations[str(pair)+'-Bader-MAE'] = 0
        
        deviations[str(pair)+'-MeanDist'] =0
        deviations[str(pair)+'-MeanDist-MAE'] = 0
                                  
    
    all_data_values.append([pair[0],
                            accuracies[str(pair)+'-Coord'],
                            deviations[str(pair)+'-Coord'],
                            accuracies[str(pair)+'-Coord-F1'],
                            deviations[str(pair)+'-Coord-F1'],
                            accuracies[str(pair)+'-GuessMode'],
                            accuracies[str(pair)+'-Bader'],
                            deviations[str(pair)+'-Bader'],
                            accuracies[str(pair)+'-Bader-MAE'],
                            deviations[str(pair)+'-Bader-MAE'],
                            accuracies[str(pair)+'-MeanDist'],
                            deviations[str(pair)+'-MeanDist'],
                            accuracies[str(pair)+'-MeanDist-MAE'],
                            deviations[str(pair)+'-MeanDist-MAE']]
                          )
    all_data_values_nn.append([pair[0],
                            accuracies_nn[str(pair)+'-Coord'],
                            deviations_nn[str(pair)+'-Coord'],
                            accuracies_nn[str(pair)+'-Coord-F1'],
                            deviations_nn[str(pair)+'-Coord-F1'],
                            accuracies_nn[str(pair)+'-GuessMode'],
                            accuracies_nn[str(pair)+'-Bader'],
                            deviations_nn[str(pair)+'-Bader'],
                            accuracies_nn[str(pair)+'-Bader-MAE'],
                            deviations_nn[str(pair)+'-Bader-MAE'],
                            accuracies_nn[str(pair)+'-MeanDist'],
                            deviations_nn[str(pair)+'-MeanDist'],
                            accuracies_nn[str(pair)+'-MeanDist-MAE'],
                            deviations_nn[str(pair)+'-MeanDist-MAE']]
                          )
        


# # Generate Performance Table

# In[ ]:


headers=['Material', 
         'Coord Baseline', 
         'Coord Acc.',
         'Coord F1 (4)','Coord F1 (5)','Coord F1 (6)',
         'Bader $R^2$', 
         'Bader MAE', 
         'Mean NN $R^2$',
         'Mean NN-MAE',]

### Random Forestmodel
f = open(figure_write_folder+'/pointwise_table_{}.csv'.format(norm_str),'w')
print(str(headers).strip('[').strip(']').replace("'",""))
f.write(str(headers).strip('[').strip(']').replace("'","")+'\n')
avgs = [0 for _ in range(len(headers))]

for pair in target_elements_groups:
    i=1
    
    elt = pair[0]
    
    the_str = elt+','
    
    the_str += "%.2f" %accuracies[str(pair)+'-GuessMode'] +','
    avgs[i] += accuracies[str(pair)+'-GuessMode']; i+=1;
    
    the_str += "%.2f" %accuracies[str(pair)+'-Coord'] +' $\pm$ '
    the_str += "%.2f" %deviations[str(pair)+'-Coord']  +','
    avgs[i] += accuracies[str(pair)+'-Coord']; i+=1;
    
    the_str += "%.2f" %accuracies[str(pair)+'-Coord-F1'][0] + ' $ \pm$ '
    the_str += "%.2f" %deviations[str(pair)+'-Coord-F1'][0] + ', '
    avgs[i] += accuracies[str(pair)+'-Coord-F1'][0]; i+=1;

    
    the_str += "%.2f" %accuracies[str(pair)+'-Coord-F1'][1] + ' $ \pm$ '
    the_str += "%.2f" %deviations[str(pair)+'-Coord-F1'][1] + ', '
    avgs[i] += accuracies[str(pair)+'-Coord-F1'][1]; i+=1;
    
    the_str += "%.2f" %accuracies[str(pair)+'-Coord-F1'][2] + ' $ \pm$ '
    the_str += "%.2f" %deviations[str(pair)+'-Coord-F1'][2] + ', '
    avgs[i] += accuracies[str(pair)+'-Coord-F1'][2]; i+=1;

    
    the_str += "%.2f" %accuracies[str(pair)+'-Bader'] +' $\pm$'
    the_str += "%.2f" %deviations[str(pair)+'-Bader'] +','
    avgs[i] += accuracies[str(pair)+'-Bader']; i+=1;

    
    the_str += "%.3f" %accuracies[str(pair)+'-Bader-MAE'] +' $\pm$'
    the_str += "%.3f" %deviations[str(pair)+'-Bader-MAE'] +' , '
    avgs[i] += accuracies[str(pair)+'-Bader-MAE']; i+=1;


    the_str += "%.2f" %accuracies[str(pair)+'-MeanDist'] +' $\pm$'
    the_str += "%.2f" %deviations[str(pair)+'-MeanDist'] +','
    avgs[i] += accuracies[str(pair)+'-MeanDist']; i+=1;

    
    the_str += "%.3f" %accuracies[str(pair)+'-MeanDist-MAE'] +' $\pm$'
    the_str += "%.3f" %deviations[str(pair)+'-MeanDist-MAE']
    avgs[i] += accuracies[str(pair)+'-MeanDist-MAE']; i+=1;

    f.write(the_str+'\n')

avgs = list(np.round(np.array(avgs)/8,2))
avgs[0]='Avgs.'
f.write(str(avgs).strip('[').strip(']'))
f.close()

### Neural Networks model
f = open(figure_write_folder+'/pointwise_table_{}_nn.csv'.format(norm_str),'w')
print(str(headers).strip('[').strip(']').replace("'",""))
f.write(str(headers).strip('[').strip(']').replace("'","")+'\n')
avgs = [0 for _ in range(len(headers))]

for pair in target_elements_groups:
    i=1
    
    elt = pair[0]
    
    the_str = elt+','
    
    the_str += "%.2f" %accuracies_nn[str(pair)+'-GuessMode'] +','
    avgs[i] += accuracies_nn[str(pair)+'-GuessMode']; i+=1;
    
    the_str += "%.2f" %accuracies_nn[str(pair)+'-Coord'] +' $\pm$ '
    the_str += "%.2f" %deviations_nn[str(pair)+'-Coord']  +','
    avgs[i] += accuracies_nn[str(pair)+'-Coord']; i+=1;
    
    the_str += "%.2f" %accuracies_nn[str(pair)+'-Coord-F1'][0] + ' $ \pm$ '
    the_str += "%.2f" %deviations_nn[str(pair)+'-Coord-F1'][0] + ', '
    avgs[i] += accuracies_nn[str(pair)+'-Coord-F1'][0]; i+=1;

    
    the_str += "%.2f" %accuracies_nn[str(pair)+'-Coord-F1'][1] + ' $ \pm$ '
    the_str += "%.2f" %deviations_nn[str(pair)+'-Coord-F1'][1] + ', '
    avgs[i] += accuracies_nn[str(pair)+'-Coord-F1'][1]; i+=1;
    
    the_str += "%.2f" %accuracies_nn[str(pair)+'-Coord-F1'][2] + ' $ \pm$ '
    the_str += "%.2f" %deviations_nn[str(pair)+'-Coord-F1'][2] + ', '
    avgs[i] += accuracies_nn[str(pair)+'-Coord-F1'][2]; i+=1;

    
    the_str += "%.2f" %accuracies_nn[str(pair)+'-Bader'] +' $\pm$'
    the_str += "%.2f" %deviations_nn[str(pair)+'-Bader'] +','
    avgs[i] += accuracies_nn[str(pair)+'-Bader']; i+=1;

    
    the_str += "%.3f" %accuracies_nn[str(pair)+'-Bader-MAE'] +' $\pm$'
    the_str += "%.3f" %deviations_nn[str(pair)+'-Bader-MAE'] +' , '
    avgs[i] += accuracies_nn[str(pair)+'-Bader-MAE']; i+=1;


    the_str += "%.2f" %accuracies_nn[str(pair)+'-MeanDist'] +' $\pm$'
    the_str += "%.2f" %deviations_nn[str(pair)+'-MeanDist'] +','
    avgs[i] += accuracies_nn[str(pair)+'-MeanDist']; i+=1;

    
    the_str += "%.3f" %accuracies_nn[str(pair)+'-MeanDist-MAE'] +' $\pm$'
    the_str += "%.3f" %deviations_nn[str(pair)+'-MeanDist-MAE']
    avgs[i] += accuracies_nn[str(pair)+'-MeanDist-MAE']; i+=1;

    f.write(the_str+'\n')

avgs = list(np.round(np.array(avgs)/8,2))
avgs[0]='Avgs.'
f.write(str(avgs).strip('[').strip(']'))
f.close()


# ## Size of training data

# In[ ]:


data_size = []
for pair in tqdm(target_elements_groups, ncols=90, desc='target_elements_groups'):
    #############################################
    # COORDINATION
    #############################################
    yc_train = ttc_by_pair[pair]['train_y'] 
    yc_train = y_augmentation(yc_train, delta=0.00)

    #############################################
    # BADER
    #############################################
    yb_train =  ttb_by_pair[pair]['train_y']
    yb_train = y_augmentation(yb_train, delta=0.02)
    
    ##############################
    #   MD PART
    ##############################
    ymd_train = np.array(ttmd_by_pair[pair]['train_y'])
    ymd_train = y_augmentation(ymd_train, delta=0.02)
    
    data_size.append([len(yc_train), len(yb_train), len(ymd_train)])

data_size_df = pandas.DataFrame(data_size)
data_size_df.columns = ['Size Coord', 'Size Bader', 'Size Mean NN']
data_size_df


# ## Performance Table: Random Forest

# In[ ]:


rf_df = pandas.read_csv(figure_write_folder+'/pointwise_table_{}.csv'.format(norm_str))
pandas.concat([rf_df, data_size_df], axis=1)


# ## Performance Table: Neural Networks

# In[ ]:


cnn_df = pandas.read_csv(figure_write_folder+'/pointwise_table_{}_nn.csv'.format(norm_str))
pandas.concat([cnn_df, data_size_df], axis=1)


# # Generate Accumulated Figures of Merit
# ## for Pointwise  models 

# ### Random Forest

# In[ ]:


plt.clf()
plt.figure(dpi=300)
for pj, pair in enumerate(target_elements_groups):
    
    
    plt.bar(pj+.5,accuracies[str(pair)+'-Coord']/100,color=colors_by_pair[pair],width=1,alpha=.7,
           label=str(pair[0]),
           yerr = deviations[str(pair)+'-Coord']/100,capsize=1.5)
    plt.bar(pj+.5,accuracies[str(pair)+'-GuessMode']/100,color='lightgrey',width=1,alpha=1)
    plt.bar(pj+.5+8,accuracies[str(pair)+'-Coord-F1'][0]/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr = deviations[str(pair)+'-Coord-F1'][0]/100, capsize=1.5)
    
    plt.bar(pj+.5+2*8,accuracies[str(pair)+'-Coord-F1'][1]/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr = deviations[str(pair)+'-Coord-F1'][1]/100,capsize=1.5)
    
    plt.bar(pj+.5+3*8,accuracies[str(pair)+'-Coord-F1'][2]/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr = deviations[str(pair)+'-Coord-F1'][2]/100,capsize=1.5)
    
    plt.bar(pj+.5+4*8,accuracies[str(pair)+'-MeanDist']/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr =  deviations[str(pair)+'-MeanDist']/100,capsize=1.5)
    
    plt.bar(pj+.5+5*8,accuracies[str(pair)+'-Bader']/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr =  deviations[str(pair)+'-Bader']/100,capsize=1.5)
    
    
for i in [0,8,16,24,32,40]:
    plt.axvline(i,color='black',lw=1,ls='-')

for i in [.2,.4,.6,.8]:
    plt.axhline(i,color='gray',ls='--',alpha=.1)

plt.xticks([8*i+4  for i in range(0,6)],
                labels=['Coordination\n Number (CN)\n Accuracy', 'F1\n(CN=4)', 'F1\n(CN=5)', 'F1\n(CN=6)',
                                      'Mean NN.\nDist. $R^2$','Bader\n$R^2$'],)
plt.xticks()
plt.xlim(0,6*7+6)
plt.ylim(-0.05,1.0000001)
plt.ylabel('Figure of Merit')


plt.title("RF: Accumulated Figures of Merit\n(Pointwise)")

plt.legend(loc='lower center',ncol=2,framealpha=.95)
plt.savefig(f'{figure_write_folder}/{norm_str}_all_perf.png',format='png',dpi=300,bbox_inches='tight',transprent=True)


# ### Convolutional Neural Networks

# In[ ]:


plt.clf()
plt.figure(dpi=300)
for pj, pair in enumerate(target_elements_groups):
    
    
    
    plt.bar(pj+.5,accuracies_nn[str(pair)+'-Coord']/100,color=colors_by_pair[pair],width=1,alpha=.7,
           label=str(pair[0]),
           yerr = deviations_nn[str(pair)+'-Coord']/100,capsize=1.5)
    plt.bar(pj+.5,accuracies_nn[str(pair)+'-GuessMode']/100,color='lightgrey',width=1,alpha=1)
    plt.bar(pj+.5+8,accuracies_nn[str(pair)+'-Coord-F1'][0]/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr = deviations_nn[str(pair)+'-Coord-F1'][0]/100, capsize=1.5)
    
    plt.bar(pj+.5+2*8,accuracies_nn[str(pair)+'-Coord-F1'][1]/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr = deviations_nn[str(pair)+'-Coord-F1'][1]/100,capsize=1.5)
    
    plt.bar(pj+.5+3*8,accuracies_nn[str(pair)+'-Coord-F1'][2]/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr = deviations_nn[str(pair)+'-Coord-F1'][2]/100,capsize=1.5)
    
    plt.bar(pj+.5+4*8,accuracies_nn[str(pair)+'-MeanDist']/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr =  deviations_nn[str(pair)+'-MeanDist']/100,capsize=1.5)
    
    plt.bar(pj+.5+5*8,accuracies_nn[str(pair)+'-Bader']/100,color=colors_by_pair[pair],width=1,alpha=.7,
           yerr =  deviations_nn[str(pair)+'-Bader']/100,capsize=1.5)
    
    
    
    
for i in [0,8,16,24,32,40]:
    plt.axvline(i,color='black',lw=1,ls='-')

for i in [.2,.4,.6,.8]:
    plt.axhline(i,color='gray',ls='--',alpha=.1)

plt.xticks([8*i+4  for i in range(0,6)],
                labels=['Coordination\n Number (CN)\n Accuracy', 'F1\n(CN=4)', 'F1\n(CN=5)', 'F1\n(CN=6)',
                                      'Mean NN.\nDist. $R^2$','Bader\n$R^2$'],)
plt.xticks()
plt.xlim(-0,6*7+6)
plt.ylim(-0.05,1.0000001)
plt.ylabel('Figure of Merit')


plt.title("NN: Accumulated Figures of Merit\n(Pointwise)")

plt.legend(loc='lower center',ncol=2,framealpha=.95)
plt.savefig(f'{figure_write_folder}/{norm_str}_all_perf_nn.png',format='png',dpi=300,bbox_inches='tight',transprent=True)


# # Performance of MD Fitting- Uniparity
# ### Random Forest

# In[ ]:



pair_to_icon={'Ti':"o",
              'V':'v',
              'Cr':'^',
              'Mn':"s",
              'Fe':"P",
              'Co':"h",
             'Ni':'D',
              'Cu':'p'}
all_min = 100
all_max = 0
plt.clf()
plt.figure(figsize=(4,4),dpi=300)
for pair in target_elements_groups:
    number = len(md_perf_by_pair[pair[0]+'-guesses'])
    all_min = min(all_min, min(md_perf_by_pair[pair[0]+'-labels']),min(md_perf_by_pair[pair[0]+'-guesses']))
    all_max = max(all_max, max(md_perf_by_pair[pair[0]+'-labels']),max(md_perf_by_pair[pair[0]+'-guesses']))

    plt.scatter(md_perf_by_pair[pair[0]+'-labels'][0],
                    md_perf_by_pair[pair[0]+'-guesses'][0],
                    zorder=-1,
                    marker=pair_to_icon[pair[0]],
                    color= colors_by_pair[pair], 
                    alpha = 1,
                    label = pair[0] + " MAE: " +str(np.round((accuracies[str(pair)+'-MeanDist-MAE']),3)))
    #print(number, len(md_perf_by_pair[pair[0]+'-labels']))
    for i in range(number)[1:]:
        plt.scatter(md_perf_by_pair[pair[0]+'-labels'][i],
                    md_perf_by_pair[pair[0]+'-guesses'][i],
                    zorder=np.random.uniform(0,1),
                    marker=pair_to_icon[pair[0]],
                    color= colors_by_pair[pair], 
                    alpha = 100/number)

plt.plot((all_min,all_max),(all_min,all_max),color='black',ls='--')
plt.legend(fontsize=8)
plt.title("RF: Mean Nearest-Neighbor Distance\nRegression Performance")
plt.xlabel("True Distance ($\AA$)")
plt.ylabel("Predicted Distance ($\AA$)")

plt.savefig(f'{figure_write_folder}/{norm_str}_md_uniparity.png',format='png',dpi=300,bbox_inches='tight')

plt.show()


# ### Convolutional Neural Networks

# In[ ]:


pair_to_icon={'Ti':"o",
              'V':'v',
              'Cr':'^',
              'Mn':"s",
              'Fe':"P",
              'Co':"h",
             'Ni':'D',
              'Cu':'p'}
all_min = 100
all_max = 0
plt.clf()
plt.figure(figsize=(4,4),dpi=300)
for pair in target_elements_groups:
    number = len(md_perf_by_pair_nn[pair[0]+'-guesses'])
    all_min = min(all_min, min(md_perf_by_pair[pair[0]+'-labels']),min(md_perf_by_pair_nn[pair[0]+'-guesses']))
    all_max = max(all_max, max(md_perf_by_pair[pair[0]+'-labels']),max(md_perf_by_pair_nn[pair[0]+'-guesses']))

    plt.scatter(md_perf_by_pair[pair[0]+'-labels'][0],
                    md_perf_by_pair_nn[pair[0]+'-guesses'][0],
                    zorder=-1,
                    marker=pair_to_icon[pair[0]],
                    color= colors_by_pair[pair], 
                    alpha = 1,
                    label = pair[0] + " MAE: " +str(np.round((accuracies_nn[str(pair)+'-MeanDist-MAE']),3)))
    #print(number, len(md_perf_by_pair[pair[0]+'-labels']))
    for i in range(number)[1:]:
        plt.scatter(md_perf_by_pair[pair[0]+'-labels'][i],
                    md_perf_by_pair_nn[pair[0]+'-guesses'][i],
                    zorder=np.random.uniform(0,1),
                    marker=pair_to_icon[pair[0]],
                    color= colors_by_pair[pair], 
                    alpha = 100/number)

plt.plot((all_min,all_max),(all_min,all_max),color='black',ls='--')
plt.legend(fontsize=8)
plt.title("NN: Mean Nearest-Neighbor Distance\nRegression Performance")
plt.xlabel("True Distance ($\AA$)")
plt.ylabel("Predicted Distance ($\AA$)")

plt.savefig(f'{figure_write_folder}/{norm_str}_md_uniparity_nn.png',format='png',dpi=300,bbox_inches='tight')

plt.show()


# ## Bader Performance
# ### Random Forest

# In[ ]:



pair_to_icon={'Ti':"o",
              'V':'v',
              'Cr':'^',
              'Mn':"s",
              'Fe':"P",
              'Co':"h",
             'Ni':'D',
              'Cu':'p'}
all_min = 100
all_max = 0
plt.clf()
plt.figure(figsize=(4,4),dpi=300)
for pair in target_elements_groups:
    number = len(bader_perf_by_pair[pair[0]+'-guesses'])
    all_min = min(all_min, min(bader_perf_by_pair[pair[0]+'-labels']),min(bader_perf_by_pair[pair[0]+'-guesses']))
    all_max = max(all_max, max(bader_perf_by_pair[pair[0]+'-labels']),max(bader_perf_by_pair[pair[0]+'-guesses']))

    plt.scatter(bader_perf_by_pair[pair[0]+'-labels'][0],
                bader_perf_by_pair[pair[0]+'-guesses'][0],
                zorder=-1,
                marker=pair_to_icon[pair[0]],
                color= colors_by_pair[pair], 
                alpha = 1,
                label=pair[0]+" MAE: "
                +str(np.round(accuracies[str(pair)+'-Bader-MAE'],2))) 
                #+f"\t R$^2$:{accuracies[str(pair)+'-Bader']:.2f}") 
    #print(number, len(md_perf_by_pair[pair[0]+'-labels']))
    for i in range(number)[1:]:
        plt.scatter(bader_perf_by_pair[pair[0]+'-labels'][i],
                    bader_perf_by_pair[pair[0]+'-guesses'][i],
                    zorder=np.random.uniform(0,1),
                    marker=pair_to_icon[pair[0]],
                    color= colors_by_pair[pair], 
                    alpha = 100/number)

plt.plot((all_min,all_max),(all_min,all_max),color='black',ls='--')
plt.legend(fontsize=8,loc='best')
plt.title("RF: Bader Charge\nRegression Performance")
plt.xlabel("True Charge (e=1)")
plt.ylabel("Predicted\nCharge (e=1)")
plt.savefig(f'{figure_write_folder}/{norm_str}_bader_uniparity.png',format='png',dpi=300,bbox_inches='tight')

plt.show()


# ### Convolutional Neural Networks

# In[ ]:


pair_to_icon={'Ti':"o",
              'V':'v',
              'Cr':'^',
              'Mn':"s",
              'Fe':"P",
              'Co':"h",
             'Ni':'D',
              'Cu':'p'}
all_min = 100
all_max = 0
plt.clf()
plt.figure(figsize=(4,4),dpi=300)
for pair in target_elements_groups:
    number = len(bader_perf_by_pair_nn[pair[0]+'-guesses'])
    all_min = min(all_min, min(bader_perf_by_pair_nn[pair[0]+'-labels']),min(bader_perf_by_pair_nn[pair[0]+'-guesses']))
    all_max = max(all_max, max(bader_perf_by_pair_nn[pair[0]+'-labels']),max(bader_perf_by_pair_nn[pair[0]+'-guesses']))

    plt.scatter(bader_perf_by_pair_nn[pair[0]+'-labels'][0],
                bader_perf_by_pair_nn[pair[0]+'-guesses'][0],
                zorder=-1,
                marker=pair_to_icon[pair[0]],
                color= colors_by_pair[pair], 
                alpha = 1,
                label=pair[0]+" MAE: "
                +str(np.round(accuracies_nn[str(pair)+'-Bader-MAE'],2))) 
                #+f"\t R$^2$:{accuracies[str(pair)+'-Bader']:.2f}") 
    #print(number, len(md_perf_by_pair[pair[0]+'-labels']))
    for i in range(number)[1:]:
        plt.scatter(bader_perf_by_pair_nn[pair[0]+'-labels'][i],
                    bader_perf_by_pair_nn[pair[0]+'-guesses'][i],
                    zorder=np.random.uniform(0,1),
                    marker=pair_to_icon[pair[0]],
                    color= colors_by_pair[pair], 
                    alpha = 100/number)

plt.plot((all_min,all_max),(all_min,all_max),color='black',ls='--')
plt.legend(fontsize=8,loc='best')
plt.title("NN: Bader Charge\nRegression Performance")
plt.xlabel("True Charge (e=1)")
plt.ylabel("Predicted\nCharge (e=1)")
plt.savefig(f'{figure_write_folder}/{norm_str}_bader_uniparity_nn.png',format='png',dpi=300,bbox_inches='tight')

plt.show()


# ### Save trained models to pickle file

# ## --------------
# 
# #                     PART TWO: POLYNOMIALS

# ## Load in the Polynomial Fit Data

# # The Big Cell - Polynomialized Data

# # Polynomial Performance Table

# --------------
# # Part 3: Plotting Comparative Performance of Training & Testing, and Feature Importance
# 

# ###  Feature Rank Function Definition

# # Plot Change in Performance from Pointwise to Polynomial

# In[ ]:


col_names = cnn_df.columns


# In[ ]:


cnn_df.columns = ['NN-'+col for col in col_names]
rf_df.columns = ['RF-'+col for col in col_names]


# In[ ]:


from scipy.stats import wilcoxon
len(raw_data)
raw_data[target_elements_groups[0]]['b'][0].keys()


# In[ ]:


e = target_elements_groups[0]

def pValue_coord_acc(e):
    a = [x['cur_model_accuracies'] for x in raw_data[e]['c']]
    b = [x['nn_model_accuracies'] for x in raw_data[e]['c']]
    return {'pvalue': wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')[1], 'rf': a, 'nn':b}

def pValue_coord_f4(e):
    a = [x['cur_model_f1s'][0] for x in raw_data[e]['c']]
    b = [x['nn_model_f1s'][0] for x in raw_data[e]['c']]
    return {'pvalue': wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')[1], 'rf': a, 'nn':b}

def pValue_coord_f5(e):
    a = [x['cur_model_f1s'][1] for x in raw_data[e]['c']]
    b = [x['nn_model_f1s'][1] for x in raw_data[e]['c']]
    return {'pvalue': wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')[1], 'rf': a, 'nn':b}

def pValue_coord_f6(e):
    a = [x['cur_model_f1s'][2] for x in raw_data[e]['c']]
    b = [x['nn_model_f1s'][2] for x in raw_data[e]['c']]
    return {'pvalue': wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')[1], 'rf': a, 'nn':b}

def pValue_r2(e, d):
    a = [x['cur_model_accuracies'] for x in raw_data[e][d]]
    b = [x['nn_model_accuracies'] for x in raw_data[e][d]]
    return {'pvalue': wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')[1], 'rf': a, 'nn':b}
def pValue_mae(e, d):
    a = [x['cur_model_maes'] for x in raw_data[e][d]]
    b = [x['nn_model_maes'] for x in raw_data[e][d]]
    return {'pvalue': wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')[1], 'rf': a, 'nn':b}


# In[ ]:


# accuracy
c_acc = [pValue_coord_acc(e) for e in target_elements_groups]

# F1(4)
c_f4 = [pValue_coord_f4(e) for e in target_elements_groups]

# F1(5)
c_f5 = [pValue_coord_f5(e) for e in target_elements_groups]

# F1(6)
c_f6 = [pValue_coord_f6(e) for e in target_elements_groups]

# Bader R2
b_r2 = [pValue_r2(e, 'b') for e in target_elements_groups]

# Bader MAE
b_mae = [pValue_mae(e, 'b') for e in target_elements_groups]

# MD R2
md_r2 = [pValue_r2(e, 'md') for e in target_elements_groups]

# MD MAE
md_mae = [pValue_mae(e, 'md') for e in target_elements_groups]


# In[ ]:


pValue_df = pandas.DataFrame({'P-Value Acc':[x['pvalue'] for x in c_acc], 
                              'P-Value F4':[x['pvalue'] for x in c_f4], 
                              'P-Value F5':[x['pvalue'] for x in c_f5], 
                              'P-Value F6':[x['pvalue'] for x in c_f6], 
                              'P-Value B R2':[x['pvalue'] for x in b_r2], 
                              'P-Value B MAE':[x['pvalue'] for x in b_mae], 
                              'P-Value MD R2':[x['pvalue'] for x in md_r2], 
                              'P-Value MD MAE':[x['pvalue'] for x in md_mae]})

pValue_raw = {'P-Value Acc':c_acc, 
              'P-Value F4':c_f4, 
              'P-Value F5':c_f5, 
              'P-Value F6':c_f6, 
              'P-Value B R2':b_r2, 
              'P-Value B MAE':b_mae, 
              'P-Value MD R2':md_r2, 
              'P-Value MD MAE':md_mae}


# In[ ]:


all_df = pandas.concat(sum([[rf_df[rf_df.columns[k]], cnn_df[cnn_df.columns[k]]] for k in range(len(cnn_df.columns))], []), axis=1)

all_df = all_df.drop([all_df.columns[0], all_df.columns[2]], axis=1)


# In[ ]:


pValue_df.insert(0, 'Material', [e[0] for e in target_elements_groups])


# In[ ]:


all_df


# ## Performance Table: P-Value

# In[ ]:


pValue_df


# In[ ]:


pValue_df.to_csv(figure_write_folder+'/pValue_table_{}_nn.csv'.format(norm_str))


# In[ ]:


all_df.to_csv(figure_write_folder+'/pValue_table_all_{}_nn.csv'.format(norm_str))


# In[ ]:


import json
with open(figure_write_folder+'/pValue_table_all_{}_nn.json'.format(norm_str), "w") as outfile:  
    json.dump(pValue_raw, outfile) 


# # AutoKeras
# ## Structured Data Classification
# ### Examples

# In[ ]:


#!{sys.executable} -m pip install numpy --upgrade
#!{sys.executable} -m pip install swish-activation
#!{sys.executable}  -m pip install git+https://github.com/keras-team/keras-tuner.git
#!{sys.executable} -m pip install autokeras

import tensorflow as tf
import autokeras as ak
import pandas as pd
import numpy as np


# In[ ]:


pair = target_elements_groups[0]

    #############################################
    # COORDINATION
    #############################################
    
xc_train = ttc_by_pair[pair]['train_x'] 
yc_train = ttc_by_pair[pair]['train_y']
xc_valid = ttc_by_pair[pair]['valid_x'] 
yc_valid = ttc_by_pair[pair]['valid_y']


# In[ ]:


## autoKeras structured data classifier
nn_c = ak.StructuredDataClassifier(overwrite=True, max_trials=5)
nn_c.fit(xc_train, yc_train, epochs=300, verbose=False)


# In[ ]:


# Predict with the best model.
#predicted_y = clf.predict(xc_valid)
# Evaluate the best model with testing data.
accu = print(nn_c.evaluate(xc_valid, yc_valid)[1])
print(accu)
precision_recall_matrix([int(p) for p in sum(clf.predict(xc_valid).tolist(), [])], yc_valid, [4,5,6])


# In[ ]:


#############################################
# BADER
#############################################
xb_train =  ttb_by_pair[pair]['train_x']
yb_train =  ttb_by_pair[pair]['train_y']
xb_train = x_augmentation(xb_train, delta=0.03)
yb_train = y_augmentation(yb_train, delta=0.0)

xb_valid = ttb_by_pair[pair]['valid_x']
yb_valid = ttb_by_pair[pair]['valid_y']
xb_valid = x_augmentation(xb_valid, delta=0.03)
yb_valid = y_augmentation(yb_valid, delta=0.0)

Y = list(yb_train) + list(yb_valid)
X = list(xb_train) + list(xb_valid)
#xb_train, xb_valid, yb_train, yb_valid = train_test_split(np.array(X), np.array(Y), test_size=0.33, random_state=42)

if run_bader:
    def processInput_b(i, forest_b=forest_b, nn_b=nn_b):
        #xb_train, xb_valid, yb_train, yb_valid = train_test_split(np.array(X), np.array(Y), test_size=0.33, random_state=i+1)
        forest_b.random_state = rseed+i+1
        forest_b.fit(xb_train,yb_train)    
        nn_b.fit(xb_train, yb_train)
        nn_perm_b = permutation_importance(nn_b, xb_train, yb_train, n_repeats=2, random_state=1)

        cur_model_importances = forest_b.feature_importances_ 
        cur_model_accuracies = r2_score(yb_valid,forest_b.predict(xb_valid)) # R-Squared
        cur_model_maes = np.mean(np.abs(forest_b.predict(xb_valid) - yb_valid))
        
        nn_model_accuracies = r2_score(yb_valid,nn_b.predict(xb_valid)) # R-Squared
        nn_model_maes = np.mean(np.abs(nn_b.predict(xb_valid) - yb_valid))
        nn_model_importances = nn_perm_b.importances_mean/np.sum(nn_perm_b.importances_mean)


# In[ ]:


xb_train =  ttb_by_pair[pair]['train_x']
yb_train =  ttb_by_pair[pair]['train_y']
xb_valid = ttb_by_pair[pair]['valid_x']
yb_valid = ttb_by_pair[pair]['valid_y']

## autoKeras structured data regression
nn_b = ak.StructuredDataRegressor(overwrite=True, max_trials=5)
nn_b.fit(xb_train, yb_train, epochs=300, verbose=False)
print(nn_b.evaluate(xb_valid, yb_valid))
r2_score(yb_valid,nn_b.predict(xb_valid))
np.mean(np.abs(nn_b.predict(xb_valid) - yb_valid))


# In[ ]:





# In[ ]:


r2_score(yb_valid,nn_b.predict(xb_valid))


# In[ ]:





# In[ ]:




