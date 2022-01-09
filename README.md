# PRÀCTICA KAGGLE APC UAB 2021-22 
# Malicious and Benign Websites Classification
#### Author: Mireia Fernández Fernández
#### DATASET (Kaggle): [Malicious and Benign Websites Dataset](https://www.kaggle.com/xwolf12/malicious-and-benign-websites)

## **Brief summary of the data**
The data used for this project has been extracted from the Kaggle website linked above and, as it has been described there, it consists of a compilation of attributes obtained via Python scripts in order to analyse further URL links to conclude if a website is malicious or not.

Said dataset, with 1781 unique samples and a total of 21 different attributes, has the following types of data: 67% of it is numerical, 33% of it is categorical and none of it has been normalized yet (this excludes, obviously, the target objective since it is actually a binary attrubute with only 0s to represent benign websites and 1s to represent malicious websites).

## **Objective**
The main goal of the project is to find the best model to classify various websites, and conclude wether if they are malicious or not. And while it will be used both supervised and unsupervised learning models, the work done is going to be focused on the first mentioned.

## **Overall testing**
Given that the number of samples is limited and it is not possible to obtain new ones easily. They have been separated into two different sets: training and testing, containing a 75% and a 25% respectively. In the future, if the dataset is expanded, this could be modified to fit better the new amount of data adquired.

## **Preprocessing** 
At first glance, it is noticeable that the data has a lot of _NaN_ and _None_ values on almost half of the attributes. For instance, the column _CONTENT_LENGTH_ has almost 46% of its samples as null values. Still, it is possible to ask for a correlation matrix of the attributes with the target one to see if there is any pattern we should be aware of: 

<p align="center">
<img src="https://github.com/mireia-fdezfdez/Malicious-and-Benign-Websites-Classification/blob/main/figures/initial_correlations.png?raw=true"width="500" />
</p>

Since there is nothing that is highly correlated to _Type_, except itself, we then proceed to clean the data, correct any spelling mistakes and fill all the blank spaces in order to obtain a new version that can be used to classify the data. Moreover, we have used a library to perform a One Hot encoding in some of the categorical attributes, as well as a Label encoding for other ones.

The result is a dataset that contains 1773 samples and 310 attributes, whoms correlations look like this: 

<p align="center">
<img src="https://github.com/mireia-fdezfdez/Malicious-and-Benign-Websites-Classification/blob/main/figures/standardized_correlations.png?raw=true"width="500" />
</p>

Given that this is a lot of attributes with little information about the one we want to predict, we apply a PC Analysis to reduce its dimensionality and, hopefully, get a new dataset with less attributes but that contains the same, or almost the same, information. After running the code shown in the _preprocessing_ notebook, we end up with the following attributes (242 principal components) and the following correlations with _Type_:

<p align="center">
<img src="https://github.com/mireia-fdezfdez/Malicious-and-Benign-Websites-Classification/blob/main/figures/pca_correlations.png?raw=true"width="300" />
</p>

## **Models Used**

| **Model**        | **Hyperparametres**  | **Accuracy**  | **Time** |
| :-------------: |:-------------:| :-----:| :-----:|
| Logistic Regressor | C = 2.8942661247167516 | 99.77477477477478 % |  0.9949207305908203 (ms) |
| KNN | K = 2 | 96.84684684684684 % |  191.48635864257812 (ms) |
| SVM | <ul><li>**C**= 78.96522868499724</li><li>**Kernel**= rbf</li></ul> | 99.54954954954955 % |  30.94768524169922 (ms) |
| SVM | <ul><li>**C**= 10</li><li>**Kernel**= linear </li></ul> | 100.0 % |  8.983135223388672 (ms) |
| Decision Tree | - | 98.87387387387388 % |  0.9963512420654297 (ms) |
| Naïve Gaussian Bayes | - | 32.207207207207205 % |  2.9926300048828125 (ms) |
| Random Forest | Trees = 111 | Acc: 99.77477477477478 % | Execution time: 0.010969877243041992 (ms) |
| Bagging Classifier | <ul><li>**Base Estimator**= 2-NN </li><li>**N Estimators**= 10 </li></ul> | 99.32432432432432 % |  240.60320854187012 (ms) |
| Bagging Classifier | <ul><li>**Base Estimator**= Logistic Regressor </li><li>**N Estimators**= 10 </li></ul> | 100.0 % |  2.9430389404296875 (ms) |
| Bagging Classifier | <ul><li>**Base Estimator**= SVC('rbf') </li><li>**N Estimators**= 10 </li></ul> | 99.32432432432432 % |  272.60327339172363 (ms) |
| Adaptive Boosting | N Estimators = 500 | 99.54954954954955 % |  155.61890602111816 (ms) |
| Gradient Boosting | <ul><li>**N Estimators**= 500 </li><li>**Maximum Depth**= 5 </li></ul> | 99.77477477477478 % |  2.994537353515625 (ms) |
| K-Means | N Clusters = 2 | 15.54054054054054 % |  0.9984970092773438 (ms) |
| K-Means | N Clusters = 3 | 83.1081081081081 % |  0.9970664978027344 (ms) |
| Spectral K-Means | N Clusters = 2 | 88.51351351351352 % |  9069.884061813354 (ms) |

## **Demo**
In order to try out and understand how the code works, a demo can be executed using the following command:
`python demo/demo.py`

## **Conclusions**
-

## **Concepts to be worked on in the future**
-
