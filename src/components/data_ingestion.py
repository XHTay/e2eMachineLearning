import os
import sys

## To solve src not found 
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))

from src.exception import CustomException # Error message when reading fails
from src.logger import logging # Logging 
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # New library in python 3.9



# Inputs required
# With the dataclass decorator, we don't need init to declare class variables, good for classes where we only define variables and not methods
@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts', "data.csv") 
    train_data_path: str=os.path.join('artifacts', "train.csv") # All outputs will be saved in artifacts
    test_data_path: str=os.path.join('artifacts', "test.csv") 

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() # All the 3 paths will be saved into this ingestion_config

    # Start reading data from source
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv') # Note: We are at working directory not component
            logging.info('Dataset read as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # if directory exists, it will not delete existing.

            # Save dataset into raw/data.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) 

            # Train test split
            logging.info("Train Test Split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

             # Save train and tes set into raw/data.csv
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            # Return these path so we can easily access the data
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestor = DataIngestion()
    data_ingestor.initiate_data_ingestion()
    train_path, test_path = data_ingestor.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path)
