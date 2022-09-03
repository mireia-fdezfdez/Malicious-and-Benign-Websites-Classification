# Malicious and Benign Websites Classification
#### Author: Mireia Fernández Fernández
#### DATASET (Kaggle): [Malicious and Benign Websites Dataset](https://www.kaggle.com/xwolf12/malicious-and-benign-websites)

## *Introduction*
This was originally a college project that I worked on earlier in my Bachelor's Degree in Computational Mathematics and Data Analytics. Using it as a base, the goal is to improve it and transform the code into the complete version of the project that I envisioned, but could not achieve, when I was first presented to it.

## **Brief summary of the data**
The data used for this project has been extracted from the Kaggle website linked above and, as it has been described there, it consists of a compilation of attributes obtained via Python scripts in order to analyse further URL links to conclude if a website is malicious or not.

Said dataset, with 1781 unique samples and a total of 21 different attributes, has the following types of data: 67% of it is numerical, 33% of it is categorical and none of it has been normalized yet (this excludes, obviously, the target objective since it is actually a binary attrubute with only 0s to represent benign websites and 1s to represent malicious websites).

## **Objective**
The main goal of the project is to find the best model to classify various websites, and conclude wether if they are malicious or not by using supervised learning models.

## **Overall testing**
Given that the number of samples is limited and it is not possible to obtain new ones easily. They have been separated into two different sets: training and testing, containing a 75% and a 25% respectively. In the future, if the dataset is expanded, this could be modified to fit better the new amount of data adquired.

## **Preprocessing** 
At first glance, it is noticeable that the data has a lot of _NaN_ and _None_ values on almost half of the attributes. For instance, the column _CONTENT_LENGTH_ has almost 46% of its samples as null values. Therefore, we proceed to clean the data, correct any spelling mistakes and fill all the blank spaces in order to obtain a new version that can be used to classify the data. Moreover, we have used the sklearn Label Encoder library to encode into numbers the categorical attributes we had.

The result is a dataset that contains 1773 samples and 21 attributes, whoms correlations look like this: 

<p align="center">
<img src="https://github.com/mireia-fdezfdez/Malicious-and-Benign-Websites-Classification/blob/main/figures/corr_dataset.png?raw=true"width="400" />
</p>

Given that 21 is still a high number of dimensions, and their correlations do not seem substancial enough, we apply a PC Analysis to reduce its dimensionality and, hopefully, get a new dataset with less attributes but that contains the same, or almost the same, information. After running the code shown in the _preprocessing_ notebook, we end up with the following information of the analysis:

Firstly, the scree plot shows us that all the components after the 13rd do not provide any useful information. And then, secondly, we can see in the plot of the cumulative explained variance that chosing a threshold of the _95%_ we can stick to only 10 components.

<p align="center">
<img src="https://github.com/mireia-fdezfdez/Malicious-and-Benign-Websites-Classification/blob/main/figures/scree_plot.png?raw=true"width="300" />
</p>

<p align="center">
<img src="https://github.com/mireia-fdezfdez/Malicious-and-Benign-Websites-Classification/blob/main/figures/expl_variance.png?raw=true"width="300" />
</p>

And thus, we end up with a dataset of only 10 dimensions that reduced the number of attributes by half from its originaly data.

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

## **Demo**
In order to try out and understand how the code works, a demo can be executed using the following command:
`python demo/demo.py`

## **Conclusions**
Overall, the majority of models have returned a perfect score, with a small variance in its output. As it can be seen the the table above the best models found are the _Baggging Classifier_ using a _Logistic regressor_ as its base and 10 independent estimators; on the other hand, another perfect accuracy has been obtained with a much simpler model, it being a _SVC_ using a _linear_ kernel and 10 as the hyperparameter _C_. All take a rather small time to both fit and predict the data so until there is a bigger amount of samples there is no objective winner to which is the best one.

## **Concepts to be worked on in the future**
Finally, this is a brief list of work that could be done to improve this project:
- It came to my attention that some people had thought of implementing neuronal networks but since my knowledge in the matter is limited I have not been able to implement one yet.
- As it could be suggested by the class that I used to contain the data, it would be interesting to analyze the behaviour of the models using only the standardized data instead of the one obtained by the PCA, or even without standardization.
- In the same way, it would be interesting to analyze the behaviour if the dataset is cleaned and filled the means of the values instead of 0s.
- Lastly, to improve the understanding of each model it would be ideal to perform analysis over different scores such as the _recall_ or the _F1_ score.
