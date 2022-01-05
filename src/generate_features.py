# -*- coding: utf-8 -*-
"""
@author: Mireia Fernández Fernández
"""
from libraries import *

class Data():
    """
    A class that is used to clean and transform data stored in a pandas dataframe.

    Attributes
    ----------
    X : object pandas.DataFrame
        An object that contains the columns with the attributes of the dataset 
        that will be used to generate features
    s_X : object pandas.DataFrame
        An object that contains the columns with the standardized attributes of 
        the dataset that will be used to generate features
    pca_X : object pandas.DataFrame
        An object that contains the columns with the principal components that 
        stand for the dimensional reduction of the dataset
    target : object pandas.Series
        An object array like that contains the target attribute values
    cleaned : bool
        A variable that specifies if the data has been cleaned or not
    app_pca : bool
        A variable that specifies if the data has gone through a dimensional
        reduction using a principal component analysis or not

    Methods
    -------
    clean_data(fill_zeros, bad_rows, bad_cols, col_changes, sample_changes,
               separate, str_to_num)
        Procedure that cleans the dataset and converts all the attributes to 
        numerical, in order to be able to predict or classify with it
    scale_data()
        Procedure that standardizes the data in X 
    pca_procedure(threshold_var)
        Procedure that applies a principal component analysis and reduces the 
        dimension of the dataset guided by an amount of explained variance 
        predetermined
    """
    
    def __init__(self, name, target):
        """
        Parameters
        ----------
        name : str
            A string with the name of the csv file that will be used as dataset
        target : str
            A string with the name of the variable to classify or predict
        """
        self.X, self.target = self._get_data(name, target)
        self.s_X = None
        self.pca_X = None
        self.cleaned = False
       
        
    def clean_data(self, fill_zeros = [], bad_rows = False, bad_cols = False, 
                   col_changes = False, sample_changes = [], separate = False,
                   str_to_num = False):
        """
        Procedure that cleans the dataset and converts all the attributes to 
        numerical, in order to be able to predict or classify with it.
        
        If it is not cleaned, it will go through all the steps needed to clean
        a dataset. First, it will fill all NaN values of the columns specified
        with 0s. Second, if there are any samples or attributes that need to be
        removed, it will do so. Third, it takes care of the specific value
        changes a specific sample may need. Fourth, separates into two any column
        that contains both dates and times. Fifth, uses the get_dummies function
        to change the categorical columns specified to numeric. Finally, it marks 
        itself as cleaned.
        
        Parameters
        ----------
        fill_zeros : list
            A list with the name of the columns that contain values NaN,
            which have to be filled with 0s (default is empty list)
        bad_rows : list
            A list with the id of the rows that have to be deleted, if no list
            is passed as a parameter (default is False)
        bad_cols : list
            A list with the name of the columns that have to be deleted, if no 
            list is passed as a parameter (default is False)
        col_changes : list
            A dictionary with the names of columns where some samples need 
            specific value changes (default is False)
        sample_changes : dict
            A list of dictionaries with the id of the rows and their new values
            (default is empty list)
        separate : list
            A list of lists with the name of the columns that originally contain
            a string which needs to be separated in a date and a time of the day
            (default is empty list)
        str_to_num : list
            A list that contains the names of the categorical columns that need
            to be converted into numerical attributes (default is False)
            
        """
        if self.cleaned == True:
            pass
            
        for col in fill_zeros:
            self.X[col] = self.X[col].fillna(0)
        
        if bad_rows != False:
            self.X.drop(bad_rows, inplace=True)
            self.target.drop(bad_rows, inplace=True)
        
        if bad_cols != False:
            self.X.drop(bad_cols, axis=1, inplace=True) 
        
        if col_changes != False:
            
            for col, samples in zip(col_changes, sample_changes):
                
                for new_vals, rows in samples.items():
                    if type(rows) == int:
                        rows = [rows]
                    for row in rows:
                        self.X.at[row,col] = new_vals
        
        if separate != False:
            
            for attr in separate:
                self._separate_date_time(attr[0], attr[1], attr[2])
        
        if str_to_num != False:
            self.X = pd.get_dummies(self.X, columns=str_to_num)
        
        self.cleaned = True


    def scale_data(self):
        """
        Procedure that standardizes the data in X 
        
        This function will only scale the data if it has been cleaned, and will
        do so using the sklearn function scale. The mean value will be moved to
        0 and component wise scale to unit variance.
        """
        if self.cleaned == False:
            pass
        
        self.s_X = preprocessing.scale(self.X)
            
        
    def pca_procedure(self, threshold_var = 0.95):
        """
        Procedure that applies a principal component analysis and reduces the 
        dimension of the dataset guided by an amount of explained variance 
        predetermined
        
        If it has been cleaned, it will create a PCA object using the sklearn
        library and then transform the dataset into the principal components.
        Finally, it will create a new X by calculating how many components the 
        matrix needs to have in order achieve the amount of explained variance
        determined by the parameter threshold_var.
        
        Parameters
        ----------
        threshold_var : float
            A variable that contains the threshold of explained variance that 
            the PCA has to achieve (default is 0.95)
        """
        if self.cleaned == False:
            pass
        
        pca = PCA()
        pca.fit(self.X)
        pca_data = pca.transform(self.X)
        
        ex_var = pca.explained_variance_ratio_
        
        cum_var = 0
        dim = -1
        
        while cum_var < threshold_var:
            if dim+1 == len(ex_var):
                break
            
            dim += 1
            cum_var += ex_var[dim]
        
        labels = ['PC'+str(i) for i in range(1, dim+1)]
        
        self.pca_X = pd.DataFrame(pca_data[:,:dim], columns=labels)
    
    
    def _get_data(self, name, target):

        data = pd.read_csv(name)
        y = data[target]
        data.drop([target], axis=1, inplace=True)
        
        return data, y
    
    
    def _separate_date_time(self, column, new_col_year, new_col_date):

        year = []
        hour = []
        f_hour_day = lambda a, b: a + b/60
        
        for i in self.X[column]:
            if (i != 'None' and type(i) != float):
                dyh = i.split(" ") 
                year.append(dyh[0].split("/")[2])
                hour.append(f_hour_day(*[int(j) for j in dyh[1].split(":")]))
            else:
                year.append(0)
                hour.append(0)
    
        self.X.insert(0, new_col_year, year, True)
        self.X.insert(0, new_col_date, hour, True)
        self.X.drop([column], axis=1, inplace=True)

    
    
    
    
    
    
    
    
    
    
    
    