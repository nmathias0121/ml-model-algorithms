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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

# MODEL : Linear Regression (works for continuous variables)
def linear_regression(X_train, X_test, y_train, y_test, show_columns, target_column):
    '''
    Linear Regression Model : 
    - performs regression task 
    - used for finding out the relationship between variables and outcome variable/target column
    - based on supervised learning

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    show_columns : names of columns to print in prediction file
    target_column : column name to predict

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
        process_data.create_prediction_file(X_test, show_columns, target_column, prediction)
    # if validating
    else :
        mae = metrics.mean_absolute_error(y_test, prediction)
        print("\nMean Absolute Error : ", mae)
        return prediction, mae
    
    print('   -----  END  -----   ')
    
    return prediction
    

# MODEL : Logistic Regression  (works for discrete variables)
def logistic_regression(X_train, X_test, y_train, y_test, show_columns, target_column):
    '''
    Logistic Regression Model : 
    - performs regression task 
    - used for finding out the relationship between variables and outcome variable/target column
    - models data using the sigmoid function
    - supervised classification algorithm
    - requires large training sets

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    show_columns : names of columns to print in prediction file
    target_column : column name to predict

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
        process_data.create_prediction_file(X_test, show_columns, target_column, model_predict)
    # if validating
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
    
    print('   -----  END  -----   ')
    
    return model_predict


# MODEL : Decision Tree Classifier
def decision_tree_classifier(X_train, X_test, y_train, y_test, show_columns, target_column):
    '''
    Decision Tree Classifier Model : 
    - performs classification & regression tasks
    - handles decision making automatically
    - prone to overfitting
    - can be trained on small training sets
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    show_columns : names of columns to print in prediction file
    target_column : column name to predict

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
        process_data.create_prediction_file(X_test, show_columns, target_column, model_predict)
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
        
    print('   -----  END  -----   ')
    
    return model_predict


# MODEL : Random Forest Classifier
def random_forest_classifier(X_train, X_test, y_train, y_test, show_columns, target_column, num_estimators):
    '''
    Random Forest Classifier Model : 
    - ensemble learning method that fits a number of decision tree classifiers
    - performs classification & regression tasks
    - handles decision making automatically
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    show_columns : names of columns to print in prediction file
    target_column : column name to predict
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
        process_data.create_prediction_file(X_test, show_columns, target_column, model_predict)
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
        
    print('   -----  END  -----   ')
    
    return model_predict


# MODEL : Gradient Boosting Classifier/Regressor
def gradient_boosting(X_train, X_test, y_train, y_test, show_columns, target_column, gb_type):
    '''
    Gradient Boosting Classifier/Regressor Model : 
    - ensemble of weak prediction models such as decision trees
    - performs classification & regression tasks
    - works for large & complex datasets, has good prediction speed & accuracy
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    show_columns : names of columns to print in prediction file
    target_column : column name to predict
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
            process_data.create_prediction_file(X_test, show_columns, target_column, model_predict)
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
            process_data.create_prediction_file(X_test, show_columns, target_column, model_predict)
        else :
            mae_train = mean_absolute_error(y_test, model.predict(X_train))
            print("Training Set Mean Absolute Error : %.2f" % mae_train)
            
            mae_test = mean_absolute_error(y_test, model.predict(X_test))
            print("Test Set Mean Absolute Error : %.2f" % mae_test)
        
        print('   -----  END  -----   ')
    
        return model_predict
    

# MODEL : k-Nearest Neighbors (for relatively small and low dimensional datasets)
# scale data before this function
def k_neighbors_classifier(X_train, X_test, y_train, y_test, show_columns, target_column, k, scaled_features):
    '''
    k Nearest Neighbors Classifier Model : 
    - uses proximity to make classifications or predictions about grouping individual data point
    - performs classification, regression & is non parametric
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    show_columns : names of columns to print in prediction file
    target_column : column name to predict
    k : number of neighbors
    scaled_features : list of columns names to predict for scaled dataset

    Output ->
    if no y_test , creates prediction file in current directory
    else
        print confusion matrix : table used to define performance of classification algorithm
        & classification report : shows main classification metrics
    model_predict : predictions for the dataset
    scaled_model_predict : predictions for the scaled dataset
    '''
    print('Running k-Nearest Neighbor Classifier....')
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    model_predict = model.predict(X_test)
    
    if y_test == None:
        process_data.create_prediction_file(X_test, show_columns, target_column, model_predict)
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
        
    scaled_model_predict = []
    if scaled_features:
        scaled_model_predict = model.predict(scaled_features)

        if y_test == None:
            process_data.create_prediction_file(X_test, show_columns, target_column, scaled_model_predict)
        else :
            print('\nConfusion Matrix : \n', confusion_matrix(y_test, scaled_model_predict))
            print('\nClassification Report : \n', classification_report(y_test, scaled_model_predict))

    print('   -----  END  -----   ')
    
    return model_predict, scaled_model_predict


# MODEL : Support Vector Machines (allows categorical variables)
def support_vector_classifier(X_train, X_test, y_train, y_test, show_columns, target_column):
    '''
    Support Vector Classifier Model : 
    - find a hyperplane in a multi dimensional space that distinctly classifies the data points
    - performs classification, regression & outlier detection
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    show_columns : names of columns to print in prediction file
    target_column : column name to predict

    Output ->
    if no y_test , creates prediction file in current directory
    else
        print confusion matrix : table used to define performance of classification algorithm
        & classification report : shows main classification metrics
    model_predictions : predictions for the dataset
    grid_predictions : predictions for the dataset using grid search
    '''
    print('Running Support Vector Classifier....')
    
    model = SVC()
    model.fit(X_train, y_train)
    
    model_predict = model.predict(X_test)
    
    if y_test == None:
        process_data.create_prediction_file(X_test, show_columns, target_column, model_predict)
        #pass
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
    
    # Grid Search
    hyperparameters = {'C':[10,25,50], 'gamma':[0.001,0.0001,0.00001]}
    grid = GridSearchCV(SVC(), hyperparameters)
    grid.fit(X_train, y_train)
    
    print('\nOptimal HyperParameter : ', grid.best_params_)
    
    grid_predictions = grid.predict(X_test)
    
    if y_test == None:
        process_data.create_prediction_file(X_test, show_columns, target_column, grid_predictions)
        #pass
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, grid_predictions))
        print('\nClassification Report : \n', classification_report(y_test, grid_predictions))
    
    print('   -----  END  -----   ')
    
    return model_predict, grid_predictions



# MODEL : Gaussian Naive Bayes Classifier
def gaussian_naive_bayes_classifier(X_train, X_test, y_train, y_test, show_columns, target_column):
    '''
    Gaussian Naive Bayes Classifier Model : 
    - based on Bayes theorem : probability of target class given selected features is inversely proportional to the probability of selected features & directly proportional to probabilty of target class or that of the selcted features given the target class
    - time efficient as naive bayes classifiers are faster than others
    - performs classification tasks
    - supervised learning algorithm

    Input ->
    X_train, X_test, y_train, y_test : train test split data
    show_columns : names of columns to print in prediction file
    target_column : column name to predict

    Output ->
    if no y_test , creates prediction file in current directory
    else
     print confusion matrix : table used to define performance of classification algorithm
           classification report : shows main classification metrics
    model_predict : predictions for the target column
    '''
    print('Running Gaussian Naive Bayes Classifier....')
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    model_predict = model.predict(X_test)
    
    if y_test == None:
        process_data.create_prediction_file(X_test, show_columns, target_column, model_predict)
    else :
        print('\nConfusion Matrix : \n', confusion_matrix(y_test, model_predict))
        print('\nClassification Report : \n', classification_report(y_test, model_predict))
        
    print('   -----  END  -----   ')
    
    return model_predict