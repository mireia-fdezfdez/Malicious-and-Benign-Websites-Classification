# -*- coding: utf-8 -*-
"""
@author: Mireia Fernández Fernández
"""
# utils
import time
import pickle
import numpy as np
import pandas as pd

# scikit-learn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import svm, tree, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import completeness_score


