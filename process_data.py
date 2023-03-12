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



# import dataset and print data specifications
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


# perform exploratory data analysis on dataset
def dataset_EDA(data, pairplot_columns):
    '''
    performs exploratory data analysis on dataset

    Input ->
    data : pandas dataframe
    pairplot_columns : columns to be plotted in pair plot and heatmap

    Output ->
    pair plot : 
    heatmap : correlation between pairplot columns
    '''
    # pairplot
    fig, (ax1) = plt.subplots(1)
    pair_plot = sns.pairplot(data, vars=pairplot_columns)
    pair_plot.savefig("eda.jpg") 

    # heatmap
    data_corr = data.corr()
    heat_map = sns.heatmap(data_corr, annot=True, cmap='coolwarm', ax=ax1)
    
    plt.show()


# perform data scrubbing
def dataset_scrubbing(data, scrub_type, data_columns, fill_operation='mean'):
    '''
    performs data cleaning on dataframe

    Input ->
    data : raw pandas dataframe
    scrub_type : remove colums / one hot encoding / drop / fill missing
    data_columns : columns to be scrubbed
    fill operation : mean or median or mode or number or string

    Output ->
    data : scrubbed / processed  pandas data frame
    '''
    if scrub_type == 'remove-columns':
        for column in data_columns:
            del data[column]
            
    elif scrub_type == 'one-hot-encoding':
        # drop_first : remove expendable columns (multi collinearity)
        data = pd.get_dummies(data, columns=data_columns, drop_first=True)
        
    elif scrub_type == 'drop-missing':
        data.dropna(axis=0, how='any', subset=data_columns, inplace=True)
    
    elif scrub_type == 'fill-missing':
        for column in data_columns:
            if fill_operation == 'mean':
                data[column].fillna(data[column].mean(), inplace=True)
            elif fill_operation == 'mode':
                data[column].fillna(data[column].mode(), inplace=True)
            elif fill_operation == 'number' or fill_operation == 'string':
                inp = input('Enter number or string to be filled :  ')
                data[column].fillna(inp)

    return data