from thyroid.logger import logging
from thyroid.exception import ThyroidException
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from thyroid.entity import config_entity, artifact_entity
from thyroid import utils
import os, sys

class DataIngestion:
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>'*30} Data Ingestion Initiated {'<'*30}")
            self.data_ingestion_config = data_ingestion_config 
        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_data_ingestion(self)-> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting the collection as dataframe")
            
            df:pd.DataFrame = utils.get_collection_as_dataframe(database_name=self.data_ingestion_config.database_name, 
                                                                collection_name=self.data_ingestion_config.collection_name)

            logging.info(f"Replacing the '?' with nan values")
            df.replace(to_replace="?",value=np.NAN, inplace=True)

            # creating feature store dir
            logging.info(f"Creating feature store folder if not availabel")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            # saving the data to feature store
            logging.info(f"Saving the data to feature store")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            # splitting the dataset
            logging.info(f"Splitting the dataset")
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=42)

            # creating the dataset directory
            logging.info(f"Creating the dataset directory if not availabel")
            dataset_dir = os.path.dirname(self.data_ingestion_config.test_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            # saving the train_df and test_df into dataset dir 
            logging.info(f"Saving the train_df and test_df into dataset dir")
            train_df.to_csv(path_or_buf = self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf= self.data_ingestion_config.test_file_path, index=False, header=True)

            # preparing the artifacts
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                                    feature_store_file_path=self.data_ingestion_config.feature_store_file_path, 
                                    train_file_path=self.data_ingestion_config.train_file_path, 
                                    test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data Ingestion Artifact Stored")
            return data_ingestion_artifact

        except Exception as e:
            raise ThyroidException(e, sys)

