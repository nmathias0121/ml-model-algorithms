# ml-model-algorithms
import datasets, perform exploratory data analysis, scaling &amp; different models such as linear or logistic regression, decision trees, random forests, K means, support vectors etc.

**Import Modules**
- Pandas
- Numpy
- Seaborn
- matplotlib

**Process Data** <br />
&emsp;process_data.py contains the following functions : <br />
&emsp;&emsp;get_file_names_in_dir(dir_name) : print name of files to process in directory  <br />
&emsp;&emsp;dataset_import(file_name, dataset_type) : import dataset & print description  such as data size, rows, columns, unique and null values  <br />
&emsp;&emsp;dataset_EDA(data, pairplot_columns) : pairplot, heatmap  <br />
&emsp;&emsp;dataset_scrubbing(data, scrub_type, data_columns, fill_operation) : clean data by removing or filling missing values, deal with categorical variables using one hot encoding, remove entire columns  <br />