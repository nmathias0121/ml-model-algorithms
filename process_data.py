import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# list all file names to process
def get_file_names_in_dir(dir_name = ''):
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



# Scale Data :PCA and k-means clustering
def pre_model_algorithm(df, algorithm, target_column):
    '''
    scales data using Principle Component Analysis or k-means clustering

    Input ->
    df : pandas dataframe to be scaled
    algorithm : pca or k-means-clustering
    target_column : name of column to be predicted

    Output ->
    (scaled pca) for pca algo. / (model_predict, centroids) for k-means-clustering algo. : scaled data
    '''
    # dimension reduction technique principle component analysis
    if algorithm == 'pca':
        scaler = StandardScaler()
        scaler.fit(df)
        scaled_data = scaler.transform(df)
        
        pca = PCA(n_components=2)
        pca.fit(scaled_data)
        scaled_pca = pca.transform(scaled_data)
        
        # print shape of scaled dataframe
        print('\n Scaled Dataset Shape : ', scaled_data.shape)
        # print shape of scaled PCA dataframe
        print('\n Scaled PCA Dataset Shape : ', scaled_pca.shape)
        
        # visualize output
        # state size of the plot
        plt.figure(figsize=(10,8))
        # configure the scatterplot's x and y axes as principle components 1 and 2
        # and color-coded by the variable
        plt.scatter(scaled_pca[:, 0], scaled_pca[:, 1], c=df[target_column])
        
        # state the scatterplot labels
        plt.xlabel('First Principle Component')
        plt.ylabel('Second Principle Component')
        
        return scaled_pca
        
    # Data Reduction Technique : k-means clustering
    elif algorithm == 'k-means-clustering':
        x,y = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=4, random_state=10)
        
        plt.figure(figsize=(7,5))
        plt.scatter(x[:, 0], x[:, 1])
        
        model = KMeans(n_clusters=4)
        model.fit(x)
        model_predict = model.predict(x)
        centroids = model.cluster_centers_ 
        print('\n Center Coordinates : \n', model.cluster_centers_)
        print('\n')
        
        # visualize output
        plt.figure(figsize=(7,5))
        plt.scatter(x[:, 0], x[:, 1], c=model_predict, s=50, cmap='rainbow')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=1);
        
        # scree plot
        # Using a for loop, iterate the values of k with a range of 1-10
        # and find the values of distortion for each k value
        distortions = []
        K = range(1,10)
        
        for k in K:
            model = KMeans(n_clusters=k)
            model.fit(x,y)
            distortions.append(model.inertia_)
        
        # Generate plot with k on the x-axis and distortions on the y-axis 
        # using matplotlib
        plt.figure(figsize=(16,8))
        plt.plot(K, distortions)
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.show()
        
        return model_predict, centroids
