import os, sys 
from thyroid.logger import logging
from thyroid.exception import ThyroidException
from datetime import datetime 

FILE_NAME = "thyroid.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = 'test.csv'
TRANSFORMER_OBJECT_FILE_NAME = 'transformer.pkl'
TARGET_ENCODER_OBJECT_FILE_NAME = 'target_encoder.pkl'
MODEL_FILE_NAME = 'model.pkl'

class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),'artifact',f"{datetime.now().strftime('%m_%d_%Y__%I_%M_%S')}")
        except Exception as e:
            raise ThyroidException(e, sys)
    
class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        try:
            self.database_name = 'thyroid-deases'
            self.collection_name = 'thyroid'
        
            # creating data_ingestion directory
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, 'data_ingestion')

            # creating feature store file path 
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,'feature_store',FILE_NAME)

            # creating train_file_path 
            self.train_file_path = os.path.join(self.data_ingestion_dir,'dataset',TRAIN_FILE_NAME)

            # creating test_file_path 
            self.test_file_path = os.path.join(self.data_ingestion_dir,'dataset',TEST_FILE_NAME)

            # test_size while splitting the dataset
            self.test_size = 0.2
        
        except Exception as e:
            raise ThyroidException(e, sys)

    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise ThyroidException(e, sys)    

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,'data_validation')
        self.report_file_path = os.path.join(self.data_validation_dir,'report.yaml')
        self.missing_threshold:float = 0.25
        self.base_file_path = os.path.join('hypothyroid_cleaned.csv')

class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir, "transformer", TARGET_ENCODER_OBJECT_FILE_NAME)
        self.transformed_train_path = os.path.join(self.data_transformation_dir, "transformed", TRAIN_FILE_NAME.replace("csv", "npz"))
        self.transformed_test_path = os.path.join(self.data_transformation_dir, "transformed", TEST_FILE_NAME.replace("csv", "npz"))
        self.target_encoder_path = os.path.join(self.data_transformation_dir, "target_encoder", TARGET_ENCODER_OBJECT_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir, "model", MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.overfitting_threshold = 0.1
        