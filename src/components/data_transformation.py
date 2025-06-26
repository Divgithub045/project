import sys 
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from src.uttils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_obj(self):
        try:
            numerical_features = ['writing_score','reading_score']
            categorical_feature = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]
            )
            logging.info("Pipeline Created for cat and num features")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_features),
                    ("cat_pipeline",categorical_pipeline,categorical_feature)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train Data Reading Completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_obj()
            target_col = "math_score"
            numerical_features = ['writing_score','reading_score']
            categorical_feature = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            input_col_train = train_df.drop(target_col,axis="columns")
            target_col_train = train_df[target_col]
            input_col_test = test_df.drop(target_col,axis="columns")
            target_col_test = test_df[target_col]
            input_feature_train_arr=preprocessing_obj.fit_transform(input_col_train)
            input_feature_test_arr=preprocessing_obj.transform(input_col_test)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_col_train)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_col_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path, # Saves the preprocessor object require for futher incoming test cases in future as pickle file
                obj=preprocessing_obj

            )
            logging.info("Preprocessing Done n Complete")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                # The preprocessing data is reusable and thus we save and only return the path while test and train arr are intermidiate to next step and gence returned directly
            )


        except Exception as e:
            raise CustomException(e,sys)
if __name__ =="__main__":
    train_path = os.path.join("artifacts", "train.csv") 
    test_path = os.path.join("artifacts", "test.csv")    
    data_transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformer.initiate_data_transformation(train_path, test_path)
    logging.info("Data Transformation Complete")
