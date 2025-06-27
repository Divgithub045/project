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
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decission Tree" : DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear":LinearRegression(),
                "K-N " : KNeighborsRegressor(),
                "CatBoost":CatBoostRegressor(verbose=0),
                "AdaBoost":AdaBoostRegressor(),
            }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model_score = max(model_report.values())
            best_model_index = list(model_report.values()).index(best_model_score)
            best_model = list(model_report.keys())[best_model_index]
            if best_model_score<0.6:
                raise CustomException("NO Best Model Found")
            logging.info("Best Model Found")
            save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj=models[best_model]  # Save the actual trained model
                        )

            best_model_instance = models[best_model]
            predicted = best_model_instance.predict(x_test)
            final_r2_score = r2_score(y_test, predicted)
            return final_r2_score
        except Exception as e:
            raise CustomException(e,sys)
