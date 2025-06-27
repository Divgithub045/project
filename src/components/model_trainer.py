import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.uttils import save_object
from src.uttils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Extracting Target and features from train and test array")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],)
            logging.info("Extraction Done for Model Training ")
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear":LinearRegression(),
                # "K-N " : KNeighborsRegressor(),
                "CatBoost":CatBoostRegressor(verbose=0),
                "AdaBoost":AdaBoostRegressor(),
            }
            params={
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'n_estimators': [50,100,150,200,250,300,350,400],
                    'max_depth': [None, 10, 20, 30],
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]

                },
                "Gradient Boosting":{
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    
                },
                "Linear":{},
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info("Hyper Para Meter Tuning Started")
            model_report,best_model=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
            best_model_score = max(model_report.values())
            # best_model_index = list(model_report.values()).index(best_model_score)
            # best_model = list(model_report.keys())[best_model_index]
            if best_model_score<0.6:
                raise CustomException("NO Best Model Found")
            logging.info("Best Model Found")
            save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model  # Save the actual trained model
                        )

            best_model_instance = best_model
            logging.info("Model Saved")
            predicted = best_model_instance.predict(x_test)
            final_r2_score = r2_score(y_test, predicted)
            return final_r2_score
        except Exception as e:
            raise CustomException(e,sys)
