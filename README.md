# ml-model-algorithms
import datasets, perform exploratory data analysis, scaling &amp; different models such as linear or logistic regression, decision trees, random forests, K means, support vectors etc.

**Import Modules**
install module in system :  <br />
&emsp;&emsp;"pip3 install module-name" <br />

**Process Data** <br />
&emsp;process_data.py contains the following functions : <br />
&emsp;&emsp;get_file_names_in_dir(dir_name) : print name of files to process in directory  <br />
&emsp;&emsp;dataset_import(file_name, dataset_type) : import dataset & print description  such as data size, rows, columns, unique and null values  <br />
&emsp;&emsp;dataset_EDA(data, pairplot_columns) : pairplot, heatmap  <br />
&emsp;&emsp;dataset_scrubbing(data, scrub_type, data_columns, fill_operation) : clean data by removing or filling missing values, deal with categorical variables using one hot encoding, remove entire columns  <br />
&emsp;&emsp;pre_model_algorithm(df, algorithm, target_column) : scale data using principle component analysis or k means clustering <br />
&emsp;&emsp;def split_validation(dataset, features, target_column, test_split) : split train data into train & test including the target column with desired split ratio <br />

**Run Model** <br />
&emsp;run_model.py contains the following models : <br />
&emsp;&emsp;linear_regression(X_train, X_test, y_train, y_test) : continuous predictions <br />
&emsp;&emsp;logistic_regression(X_train, X_test, y_train, y_test) : discrete predictions <br />
&emsp;&emsp;decision_tree_classifier(X_train, X_test, y_train, y_test) : both continuous & discrete predictions <br />
&emsp;&emsp;random_forest_classifier(X_train, X_test, y_train, y_test, num_estimators) : both continuous & discrete predictions <br />
&emsp;&emsp;gradient_boosting(X_train, X_test, y_train, y_test, gb_type) : regressor for continuous & classifier for discrete <br />
&emsp;&emsp;k_neighbors_classifier(X_train, X_test, y_train, y_test, k, scaled_features) : continuous, discrete, ordinal, categorical data predictions <br />
&emsp;&emsp;s   <br />


**References**
- https://www.geeksforgeeks.org/
- https://scikit-learn.org/stable/index.html
- https://www.ibm.com/topics/knn#:~:text=The%20k%2Dnearest%20neighbors%20algorithm%2C%20also%20known%20as%20KNN%20or,of%20an%20individual%20data%20point.
