# -*- coding: utf-8 -*-
"""
@author: Mireia Fernández Fernández
"""
from libraries import *
import generate_features as gf


def train_LR(X_train, y_train, C = 50, penalty = 'l2', intercept = True,
             tolerance = 0.0001, save_model = False):
    """
    Procedure that trains a LogisticRegressor model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        C : float
            A variable that contains the value of the regularization parameter 
            which the model will be initialized with (default is 50)
        penalty : str
            A variable that contains the name of the type of penalty which the 
            model will be initialized with (default is 'l2')
        intercept : bool
            A variable that informs if the model will use an intercept constant
            or not (default is True)
        tolerance : float
            A variable that contains the value of the tolerance parameter 
            which the model will be initialized with (default is 0.0001)
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    logireg = LogisticRegression(C = C, fit_intercept = intercept, 
                                 penalty = penalty, tol = tolerance) 
    logireg.fit(X_train, y_train)
    
    if (save_model):    
        pickle.dump(logireg, open('../models/'+str(C)+'lr.sav', 'wb'))


def train_KNN(X_train, y_train, num_neighs = 10, save_model = False):
    """
    Procedure that trains a KNeighborsClassifier model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        num_neighs : int
            A variable that contains the value of the number of neighbours
            which the model will be initialized with (default is 10)
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    knnreg = KNeighborsClassifier(n_neighbors = num_neighs)
    knnreg.fit(X_train, y_train)
    
    if (save_model):
        pickle.dump(knnreg, open('../models/'+str(num_neighs)+'nn.sav', 'wb'))
        
        
def train_SVC(X_train, y_train, C = 10, k = 'linear', save_model = False):
    """
    Procedure that trains a SupportVectorClassifier model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        C : float
            A variable that contains the value of the regularization parameter 
            which the model will be initialized with (default is 10)
        kernel : str
            A variable that contains the name of the type of kernel which the 
            model will be initialized with (default is 'linear')
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    svmreg = svm.SVC(C = C, kernel = k, probability = True)
    svmreg.fit(X_train, y_train)
    
    if (save_model):
        pickle.dump(svmreg, open('../models/'+str(C)+str(k)+'svc.sav', 'wb'))
        
        
def train_DT(X_train, y_train, save_model = False):
    """
    Procedure that trains a DecisionTree model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    treereg = tree.DecisionTreeClassifier()
    treereg.fit(X_train, y_train)
    
    if (save_model):
        pickle.dump(treereg, open('../models/dt.sav', 'wb'))


def train_GNB(X_train, y_train, save_model = False):
    """
    Procedure that trains a GaussianNaïveBayes model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    gnbreg = naive_bayes.GaussianNB()
    gnbreg.fit(X_train, y_train)
    
    if (save_model):
        pickle.dump(gnbreg, open('../models/ngb.sav', 'wb'))
        

def train_RF(X_train, y_train, num_trees = 50, save_model = False):
    """
    Procedure that trains a RandomForest model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        num_trees : int
            A variable that contains the value of the number of estimators
            which the model will be initialized with (default is 50)
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    rfreg = RandomForestClassifier(n_estimators = num_trees)
    rfreg.fit(X_train, y_train)
    
    if (save_model):
        pickle.dump(rfreg, open('../models/'+str(num_trees)+'rf.sav', 'wb'))
        
   
def train_BC(X_train, y_train, n = 10, model = 'KNN', save_model = False):
    """
    Procedure that trains a BaggingClassifier model
    
    It initializes the model with the parameters given and fits the data. This
    ensamble method will use simpler models with their best hyperparameters that
    have been found respectively to ensure their optimal performance. If it is 
    asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        n : int
            A variable that contains the value of the number of estimators
            which the model will be initialized with (default is 10)
        model : str
            A variable that contains the name of the type of classifier which the 
            model will be initialized with (default is 'linear')
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    
    if model == 'lr':
        reg = LogisticRegression(C = 2.8942661247167516, tol = 0.00001)
    elif model == 'knn':
        reg = KNeighborsClassifier(n_neighbors = 2)
    else:
        reg = svm.SVC(C = 78.96522868499724, kernel = 'rbf', probability = True)
        
    bagreg = BaggingClassifier(base_estimator = reg, n_estimators = n, 
                               random_state = 0)
    bagreg.fit(X_train, y_train)
    
    if (save_model):
        pickle.dump(bagreg, open('../models/'+str(model)+str(n)+'bc.sav', 'wb'))


def train_AB(X_train, y_train, num_estimators = 50, save_model = False):
    """
    Procedure that trains a AdaBoostClassifier model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        num_estimators : int
            A variable that contains the value of the number of estimators
            which the model will be initialized with (default is 50)
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    abreg = AdaBoostClassifier(n_estimators = num_estimators)
    abreg.fit(X_train, y_train)
    
    if (save_model):
        pickle.dump(abreg, open('../models/'+str(num_estimators)+'ab.sav', 'wb'))
        
        
def train_GB(X_train, y_train, num_estimators = 50, depth = 5, save_model = False):
    """
    Procedure that trains a GradientBoostingClassifier model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        num_estimators : int
            A variable that contains the value of the number of estimators
            which the model will be initialized with (default is 50)
        depth : int
            A variable that contains the value of the maximum depth which the 
            model will be initialized with (default is 5)
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    gbreg = GradientBoostingClassifier(n_estimators = num_estimators, 
                                        max_depth = depth, random_state = 0)
    gbreg.fit(X_train, y_train)
    
    if (save_model):
        pickle.dump(gbreg, open('../models/'+str(num_estimators)+'gb.sav', 'wb'))


def train_KM(X_train, num_clusters = 2, save_model = False):
    """
    Procedure that trains a KMeans model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        num_clusters : int
            A variable that contains the value of the number of clusters which
            the model will be initialized with (default is 2)
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    kmreg = KMeans(n_clusters = num_clusters)
    kmreg.fit(X_train)
    
    if (save_model):
        pickle.dump(kmreg, open('../models/'+str(num_clusters)+'km.sav', 'wb'))
        
        
def train_SKM(X_train, num_clusters = 2, save_model = False):
    """
    Procedure that trains a SpectralClustering model
    
    It initializes the model with the parameters given and fits the data. If it 
    is asked to, then it will use the pickle module to save the model.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        num_clusters : int
            A variable that contains the value of the number of clusters which
            the model will be initialized with (default is 2)
        save_model : bool
            A variable that informs if the model trained has to be saved locally
            or not (default is False)
        
    """    
    skmreg = SpectralClustering(n_clusters = num_clusters)
    skmreg.fit(X_train)
    
    if (save_model):
        pickle.dump(skmreg, open('../models/'+str(num_clusters)+'skm.sav', 'wb'))



     
