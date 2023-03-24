import process_data
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble

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
    - requires large training sets

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


# MODEL : Decision Tree Classifier
def decision_tree_classifier(X_train, X_test, y_train, y_test):
    '''
    decision tree classifier model : 
    - performs classification & regression tasks
    - handles decision making automatically
    - prone to overfitting
    - can be trained on small training sets
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data

    Output ->
    if no y_test , creates prediction file in current directory
    else
     print confusion matrix : table used to define performance of classification algorithm
           classification report : shows main classification metrics
    model_predict : predictions for the target column
    '''
    print('Running Decision Tree Classifier....')
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    model_predict = model.predict(X_test)
    
    if y_test == None:
        process_data.create_prediction_file(X_test, ['PassengerId'], 'Survived', model_predict)
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
        
    print('   -----  END  -----   ')
    
    return model_predict


# MODEL : Random Forest Classifier
def random_forest_classifier(X_train, X_test, y_train, y_test, num_estimators):
    '''
    random forest classifier model : 
    - ensemble learning method that fits a number of decision tree classifiers
    - performs classification & regression tasks
    - handles decision making automatically
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    num_estimators : number of trees in forest

    Output ->
    if no y_test , creates prediction file in current directory
    else
     print confusion matrix : table used to define performance of classification algorithm
           classification report : shows main classification metrics
    model_predict : predictions for the target column
    '''
    print('Running Random Forest Classifier....')
    
    model = RandomForestClassifier(n_estimators=num_estimators)
    model.fit(X_train, y_train)
    
    model_predict = model.predict(X_test)
    
    if y_test == None:
        process_data.create_prediction_file(X_test, ['PassengerId'], 'Survived', model_predict)
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
        
    print('   -----  END  -----   ')
    
    return model_predict


# MODEL : Gradient Boosting Classifier/Regressor
def gradient_boosting(X_train, X_test, y_train, y_test, gb_type):
    '''
    gradient boosting classifier/regressor model : 
    - ensemble of weak prediction models such as decision trees
    - performs classification & regression tasks
    - works for large & complex datasets, has good prediction speed & accuracy
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    gb_type : 'classifier' or 'regressor'

    Output ->
    if no y_test , creates prediction file in current directory
    else
    - if classifier
           print confusion matrix : table used to define performance of classification algorithm
           & classification report : shows main classification metrics
    - else if regressor
            print mean absolute error
    model_predict : predictions for the target column
    '''
    if gb_type == 'classifier':
        print('Running Gradient Boosting Classifier....')
        
        model = ensemble.GradientBoostingClassifier(n_estimators=250, learning_rate=0.1, max_depth=5, min_samples_split=4, min_samples_leaf=6, max_features=0.6, loss='deviance')
        model.fit(X_train, y_train)
        
        model_predict = model.predict(X_test)
        
        if y_test == None:
            process_data.create_prediction_file(X_test, ['PassengerId'], 'Survived', model_predict)
        else :
            print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
            print('\nClassification Report : \n', classification_report(y_test, model_predict))
            
        print('   -----  END  -----   ')
    
        return model_predict
    
    # perform one-hot-encoding on categorical variables before
    elif gb_type == 'regressor':
        print('Running Gradient Boosting Regressor....')
        
        model = ensemble.GradientBoostingRegressor(n_estimators=350, learning_rate=0.1, max_depth=5, min_samples_split=4, min_samples_leaf=6, max_features=0.6, loss='huber')
        model.fit(X_train, y_train)
        
        model_predict = model.predict(X_test)

        if y_test == None:
            process_data.create_prediction_file(X_test, ['PassengerId'], 'Survived', model_predict)
        else :
            mae_train = mean_absolute_error(y_test, model.predict(X_train))
            print("Training Set Mean Absolute Error : %.2f" % mae_train)
            
            mae_test = mean_absolute_error(y_test, model.predict(X_test))
            print("Test Set Mean Absolute Error : %.2f" % mae_test)
        
        print('   -----  END  -----   ')
    
        return model_predict