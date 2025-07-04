import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}
        best_model = None
        best_score = float('-inf')
        best_model_name = ""
        best_params = None
        for i in range(len(list(models))):
            model_name = list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs = GridSearchCV(model_name,param,cv=3,scoring='r2')
            gs.fit(x_train,y_train)
            model = gs.best_estimator_
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
            if test_model_score > best_score:
                best_score = test_model_score
                best_model = model
                best_model_name = model_name
                best_params = gs.best_params_

        print(f"Best Model: {best_model_name}")
        print(f"Best Parameters: {best_params}")
        return report,best_model
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    