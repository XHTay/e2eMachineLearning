import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer # Preprocessing Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer # Handling null values
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl") # Saving transformation pipeline into artifacts


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Create pickle files for al your preprocessor objects
    def get_data_transformer_object(self):
        '''
        This function defines our data transformation pipeline
        '''
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

            # Create pipelines
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # Since outliers exists
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), # Mode for categorical imputing
                    ("ohe", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exeption as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)

            logging.info("Train and test data read completed.")

            logging.info("Obtaining preprocessing objects.")

            preprocessing_obj = self.get_data_transformer_object()

            target = "math score"

            numerical_columns = ["writing score", "reading score"]

            feature_train_df = df_train.drop(columns=[target], axis=1)
            target_train_df = df_train[target]
            feature_test_df = df_test.drop(columns=[target], axis=1)
            target_test_df = df_test[target]

            logging.info(f"Applyying preprocessing object on training dataframe and testing dataframe.")

            feature_train_arr = preprocessing_obj.fit_transform(feature_train_df) # Fit scaler with train data
            feature_test_arr = preprocessing_obj.transform(feature_test_df) # Scale test data with scaler fitted with train data

            train_arr = np.c_[
                feature_train_arr, np.array(target_train_df)
            ]
            test_arr = np.c_[feature_test_arr, np.array(target_test_df)]

            # Create this function in utils, which stores common functions that is used in various places of the project.
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
