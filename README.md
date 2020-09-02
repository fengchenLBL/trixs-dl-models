# trixs-dl-models
## [Model](Train_Run_DL_Models.ipynb):
* [A Deep-Learning model](Train_Run_DL_Models.ipynb) based on the [Random Forest model](https://github.com/TRI-AMDD/trixs/blob/Torrisi_XANES_RF_2020/notebooks/Train_Run_Models.ipynb) from a published article: 
   * [Random forest machine learning models for interpretable X-ray absorption near-edge structure spectrum-property relationships](https://www.nature.com/articles/s41524-020-00376-6)
* Current version: DL vs RF on pointwise spectra data
  * NN model without regularization : [Train_Run_DL_Models.ipynb](Train_Run_DL_Models.ipynb)
  * NN model with regularization : [Train_Run_DL_Models_V2.ipynb](Train_Run_DL_Models.ipynb)
  * CNN model : [Train_Run_DL_Models_CNN.ipynb]

## Data:
* training data: https://data.matr.io/4/
```
wget https://s3.amazonaws.com/publications.matr.io/4/deployment/data/xanes_2019.zip
unzip xanes_2019.zip
cp -rf matrio_folder/spectral_data matrio_folder/model_data ./
```
## References:
* [https://www.nature.com/articles/s41524-020-00376-6](https://www.nature.com/articles/s41524-020-00376-6)
* [https://github.com/TRI-AMDD/trixs/blob/Torrisi_XANES_RF_2020/notebooks/Train_Run_Models.ipynb](https://github.com/TRI-AMDD/trixs/blob/Torrisi_XANES_RF_2020/notebooks/Train_Run_Models.ipynb)
* [https://data.matr.io/4/](https://data.matr.io/4/)
