# -*- coding: utf-8 -*-
"""
@author: Mireia Fernández Fernández
"""
from libraries import *
import generate_features as gf
from train_model import *
from score_model import *


#########################################################
############# CLEANING, PREPROCESSING, PCA ##############
#########################################################

# creates and object, which automatically loads the dataset
data = gf.Data("dataset.csv","Type")

# to clean the data we frist create the variables to do so
fill_w_zeros = ["CONTENT_LENGTH", "DNS_QUERY_TIMES"]
bad_rows = [104, 290, 357, 383, 1067, 1306, 1394, 1400]
bad_cols = ["URL", "WHOIS_STATEPRO"]
col_changes = ["WHOIS_COUNTRY", "WHOIS_REGDATE"]
uk_list = [1001]+data.X.index[data.X["WHOIS_COUNTRY"] == "GB"].tolist() 
wc_dict = {"SE":30, "CY":[1353,1411],"UK": uk_list,
           "RU":data.X.index[data.X["WHOIS_COUNTRY"] == "ru"].tolist()}
wr_dict = {"None":1360}
sample_changes = [wc_dict, wr_dict]
separate = [["WHOIS_REGDATE", "WHOIS_REGD_YEAR", "WHOIS_REGD_HOUR"],
            ["WHOIS_UPDATED_DATE", "WHOIS_REGUP_YEAR", "WHOIS_REGUP_HOUR"]]
str_to_num = ["CHARSET","SERVER","WHOIS_COUNTRY"]

# now we are ready to call the function
data.clean_data(fill_w_zeros,bad_rows,bad_cols,col_changes,sample_changes,
              separate,str_to_num)

# next we are going to standardize the data
data.scale_data()

# finally, we apply a pca to the standardized data
data.pca_procedure(0.9)



#########################################################
#################### TRAINING MODELS ####################
#########################################################

 # first we separate the test into two different subsets: train and test
X_train, X_test, y_train, y_test = train_test_split(data.pca_X, data.target, train_size = 0.75)

# as an example, we will train a couple of models that has performed excellently
train_SVC(X_train, y_train, C = 10, k = 'linear', save_model = True)
train_SVC(X_train, y_train, C = 78.96522868499724, k = 'rbf', save_model = True)


#########################################################
################## TESTING SOME MODELS ##################
#########################################################

# and then we can predict the output labels with the score_model library
acc, time = score_accuracy_model(X_test, y_test, '78.96522868499724rbfsvc.sav', True)
print("Model: SVM | C= 78.9652 and Kernel= rbf | "+str(100*acc), "% | ", time*1000, "(ms) |")

acc, time = score_accuracy_model(X_test, y_test, '10linearsvc.sav', True)
print("Model: SVM | C= 10.0 and Kernel= linear | "+str(100*acc), "% | ", time*1000, "(ms) |")



