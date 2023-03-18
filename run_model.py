import process_data
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# MODEL : Linear Regression (works for continuous variables)
def linear_regression(X_train, X_test, y_train, y_test):
    '''
    linear regression model : 
    - performs regression task 
    - used for finding out the relationship between variables and outcome variable/target column
    - based on supervised learning

    Input ->
    X_train, X_test, y_train, y_test : train test split data

    Output ->
    if no y_test , creates prediction file in current directory  & returns prediction list
    else
     prediction, mae : predictions for the target column, mean absolute error
    '''
    print('Running Linear Regression....')
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # find y-intercept and x coefficients
    print('\n y-intercept : ', model.intercept_)
    model_results = pd.DataFrame(model.coef_, X_train.columns, columns=['Coefficients'])
    print('\n x coefficients : \n', model_results)
    print('\n')
    
    prediction = model.predict(X_test)

    mae = 0
    # if predictions dont exist
    if y_test == None:
        process_data.create_prediction_file(X_test, ['PassengerId'], 'Survived', prediction)
    # if validating
    else :
        mae = metrics.mean_absolute_error(y_test, prediction)
        print("\nMean Absolute Error : ", mae)
        return prediction, mae
    
    print('   -----  END  -----   ')
    
    return prediction
    

# MODEL : Logistic Regression  (works for discrete variables)
def logistic_regression(X_train, X_test, y_train, y_test):
    '''
    logistic regression model : 
    - performs regression task 
    - used for finding out the relationship between variables and outcome variable/target column
    - models data using the sigmoid function
    - supervised classification algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data

    Output ->
    if no y_test , creates prediction file in current directory
    else
     print confusion matrix : table used to define performance of classification algorithm
           classification report : shows main classification metrics
    model_predict : predictions for the target column
    '''
    print('Running Logistic Regression....')
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    model_predict = model.predict(X_test)
    
    # if predictions dont exist
    if y_test == None:
        process_data.create_prediction_file(X_test, ['PassengerId'], 'Survived', model_predict)
    # if validating
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
    
    print('   -----  END  -----   ')
    
    return model_predict