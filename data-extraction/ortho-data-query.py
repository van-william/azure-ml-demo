import pandas as pd
import numpy as np
import json
import arff
from io import StringIO
import requests, zipfile
import sys                                              
import os
import urllib.request


current_path = os.getcwd()
f_zip = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'
r = requests.get(f_zip)
with open(current_path+'ortho-zip.zip', 'wb') as f:
    f.write(r.content)
with zipfile.ZipFile(current_path+'ortho-zip.zip', 'r') as zip:
    # printing all the contents of the zip file
    print('directory: \n')
    zip.printdir()
    # extracting all the files
    zip.extractall()

# Extract Binary Class of Orthopedic Data
data_dict_2C = arff.load(open(current_path+'/column_2C_weka.arff'))
data_arff_2C = data_dict_2C["data"]
attributes_2C = data_dict_2C['attributes']
features_2C = [item[0] for item in attributes_2C]
ortho_2C_df = pd.DataFrame(data = data_arff_2C, columns = features_2C)
ortho_2C_df.to_csv(path_or_buf='ortho-data-binary-class.csv', index=False)

# Extract Multi-Class of Orthopedic Data
data_dict_3C = arff.load(open(current_path+'/column_3C_weka.arff'))
data_arff_3C = data_dict_3C["data"]
attributes_3C = data_dict_3C['attributes']
features_3C = [item[0] for item in attributes_3C]
ortho_3C_df = pd.DataFrame(data = data_arff_3C, columns = features_3C)
ortho_3C_df.to_csv(path_or_buf='ortho-data-multiclass.csv', index=False)
