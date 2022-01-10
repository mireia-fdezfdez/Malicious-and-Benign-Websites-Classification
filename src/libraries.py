# -*- coding: utf-8 -*-
"""
@author: Mireia Fernández Fernández
"""

import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA
import time
import numpy as np
from sklearn import metrics
from sklearn import svm, tree, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import completeness_score


