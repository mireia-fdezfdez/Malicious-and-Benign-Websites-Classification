# -*- coding: utf-8 -*-
"""
@author: Mireia Fernández Fernández
"""
from libraries import *

def score_accuracy_model(X_test, y_test, name_model, return_time = False, 
                     predict = True):
    """
    Function that tests a said model and returns its accuracy
    
    It loads the model with help of the pickle module, predicts the targets for
    the data givent and then returns the accuracy score.
        
        Parameters
        ----------
        X_train : object pandas.Dataframe
            An object that contains the samples of the train dataset
        y_train : object pandas.Series
            An object that contains the true values of the attribute that has 
            to be predicted with X_train
        name_model : str
            A variable that contains the name of the file .sav that contains the
            name of the model that is required to predict the data given.
        return_time : bool
            A variable that informs if the function has to return the elapsed
            time of the prediction or not (default is False)
        predict : bool
            A variable that informs if the model that given has a predict method
            built in or not (default is True)
            
        Returns
        -------
        - : float
            A float varaible that contains the accuracy that the model has when
            it predicts the targets of the test samples given.
        elapsed_time : float
            A variable that contains how much time was needed to predict the 
            labels of the target attribute with the model given.
        
    """    
    model = pickle.load(open(name_model, 'rb'))
    
    if predict:
        ant = time.time()
        y_pred = model.predict(X_test)
        elapsed_time = time.time() - ant
    else:
        ant = time.time()
        y_pred = model.fit_predict(X_test)
        elapsed_time = time.time() - ant
    
    if return_time:
        return metrics.accuracy_score(y_test, y_pred), elapsed_time
    
    return metrics.accuracy_score(y_test, y_pred)  
        


