from thyroid.logger import logging
from thyroid.exception import ThyroidException
from datetime import datetime
import os, sys 
import pandas as pd 
import numpy as np 
from thyroid.entity import config_entity, artifact_entity
from scipy.stats import ks_2samp
from thyroid.config import TARGET_COLUMN, NUMERICAL_COLUMN
import yaml
from typing import Optional, List
from thyroid import utils


class DataValidation:
    def __init__(self, 
                data_validation_config:config_entity.DataValidationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>'*30} Data Validation Initiated {'<'*30}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact =data_ingestion_artifact
            self.validation_error = dict() # this line of code create empty dictonary
        
        except Exception as e:
            raise ThyroidException(e, sys)

    def drop_missing_values(self, df:pd.DataFrame, report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop columns which contains missing value more than specified threshold
        
        df : pandas dataframe
        threshold : percentage criteria to drop columns
        ===========================================================================================
        return pandas dataframe if atleast a single column is available after missing column drop else None
        """
        try:
            threshold = self.data_validation_config.missing_threshold 
            null_report = df.isna().sum()/df.shape[0]

            # selecting column which contains the null
            logging.info(f"Selecting the column name which contaion null above the threshold of {threshold}")
            drop_column_name = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_name)}")
            self.validation_error[report_key_name]=list(drop_column_name)

            df.drop(list(drop_column_name), axis=1, inplace=True)
            logging.info(f"Columns dropped")

            # return none if no column left
            if len(df.columns) == 0:
                return None
            return df 

        except Exception as e:
            raise ThyroidException(e, sys)

    def is_required_column_exist(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str) ->bool:
        try:
            # get the column name in base and current dataframe
            base_columns = base_df.columns
            current_columns = current_df.columns

            # initializing an empty list to store missing columns
            missing_column = []

            # iterate through the column in database columns 
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column {base_column} is not availabel in current dataset")
                    missing_column.append(base_column)

            # check if there are any missing columns
            if len(missing_column)>0:
                # add the missing column to validation error dictonary
                self.validation_error[report_key_name]=missing_column
                # return False to indicate the missing columns
                return False
            # return True to indicate there are no missing columns
            return True 
        except Exception as e:
            raise ThyroidException(e, sys)

    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame,column_list:List, report_key_name:str):
        try:
            # initializing an empty dictonary to store the data drift
            drift_report = dict()

            base_columns = base_df[column_list]
            current_columns = current_df[column_list]

            for base_column in base_columns:
                # get the data for corresponding column in which current dataframe
                base_data, current_data = base_df[base_column], current_df[base_column]

                # null hypothesis is that both columns are drawn from same distribution 
                same_distribution = ks_2samp(data1=base_data, data2=current_data)

                logging.info(f"Checking null hypothesis")
                # checking if null hypothesis is rejected or accepted 
                if same_distribution.pvalue > 0.05:
                    # we are accepting the null hypothesis 
                    drift_report[base_column] ={
                        "pvalue":float(same_distribution.pvalue),
                        "same_distribution":True
                    }
                else:
                    drift_report[base_column] ={
                        "pvalue":float(same_distribution),
                        "same_distribution":False
                    }
            
            # add the drift report to validation error directory
            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_data_validation(self) ->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading the dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            logging.info(f"Replacing '?' values in df")
            base_df.replace({'?':np.NAN}, inplace=True)

            logging.info(f"Dropping null values columns from the base_df")
            base_df = self.drop_missing_values(df=base_df, report_key_name="missing_values_within_base_dataset")

            logging.info(f"Reading the train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Reading the test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Drop null values column train df")
            train_df = self.drop_missing_values(df=train_df, report_key_name="missing_values_within_train_dataset")
            logging.info(f"Drop null values column test df")
            test_df = self.drop_missing_values(df=test_df, report_key_name="missing_values_within_test_dataset")

            exclude_columns = [TARGET_COLUMN]

            logging.info(f"Checking is all required columns are present in train df")
            train_df_columns_status = self.is_required_column_exist(base_df=base_df, current_df=train_df, report_key_name="missing_columns_within_train_dataset")
            logging.info(f"Checking is all required columns are present in train df")
            test_df_columns_status = self.is_required_column_exist(base_df=base_df, current_df=test_df, report_key_name="missing_columns_within_test_dataset")

            if train_df_columns_status:
                logging.info(f"As all columns are availabel in train df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df, column_list=NUMERICAL_COLUMN, report_key_name="data_drift_within_train_dataset")
            if test_df_columns_status:
                logging.info(f"As all columns are availabel in test df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df, column_list=NUMERICAL_COLUMN, report_key_name="data_drift_within_test_dataset")

            #write the report
            logging.info("Write reprt in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
            data=self.validation_error)   

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path,)
            logging.info(f"Data validation artifact stored")
            return data_validation_artifact

        except Exception as e:
            raise ThyroidException(e, sys)
