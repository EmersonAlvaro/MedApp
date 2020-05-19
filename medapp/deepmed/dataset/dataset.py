from settings import *

import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn.compose import ColumnTransformer
import numpy as np

class DataSet:

    def __init__(self):
        X, Y = self.load_dataset()

        # print(" {} \n input shape {}".format(X, X.shape))
        # print("{} \n labels in one-hot encoding shape {}".format(Y, Y.shape))
       
    def load_dataset(self):
        path = Path.joinpath(data_dir,'onehotenc_symptoms_disease.csv')
        df = pd.read_csv(path)
        
        le = LabelEncoder()
        df.disease = le.fit_transform(df.disease.values)                   
        
        # ct = ColumnTransformer([("disease", OneHotEncoder(), [1])],    remainder = 'passthrough')
        # df = ct.fit_transform(df)

        X = df.symptoms.values
        Y = df.disease.values

        bitmap_size =  len(X[0].split(' '))
        num_examples = len(X)

        X = " ".join(X)
        X = X.split(" ")
        
        for i in range(len(X)):
            X[i] = float(X[i])

        X = np.array(X)
        X = X.reshape(num_examples, bitmap_size, 1)
        Y = np.array(Y)

        return X, Y