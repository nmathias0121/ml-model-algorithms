import os 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline


# list all file names to process
def get_file_names_current_dir(dir_name = ''):
    for dirname, _, filenames in os.walk(os.getcwd() + dir_name):
        for filename in filenames:
            print(os.path.join(dirname, filename))



# import dataset and show data specifications
# dataset_type : Train or Test
def dataset_import(file_name, dataset_type) -> pd.DataFrame():
    '''
    imports dataset and provides information on data.

    Input ->
    file_name : name of file to be processed for running models
    dataset_type : Train or Test dataset

    Output ->
    data : a pandas dataframe that is accepted by model algorithms
    '''

    # import dataset
    print('*******  ' + dataset_type + ' Dataset  *********')
    data = pd.read_csv(file_name)
    
    # get info on no. of rows, columns  ,  data types of columns  ,  names of columns
    print('Shape of ' + dataset_type + ' Dataset : ', data.shape)
    print('\nData types  : \n', data.dtypes)
    print('\n' + dataset_type + ' Dataset Columns : \n', data.columns)

    #print('\nShort Description : \n', data.describe)

    # print unique values of each column
    print('\nUnique Values of Columns : \n')
    for col in data.columns:
        print(col + ' : \n\t')
        print(data[col].unique())

    # show number of null values for each column in dataframe
    print('\nNull Values in ' + dataset_type +' Dataset :\n', data.isnull().sum())
    
    return data