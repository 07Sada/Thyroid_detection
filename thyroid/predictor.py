import os, sys
from thyroid.entity.config_entity import TRANSFORMER_OBJECT_FILE_NAME, FILE_NAME, MODEL_FILE_NAME, TARGET_ENCODER_OBJECT_FILE_NAME
from glob import glob
from typing import Optional
from thyroid.exception import ThyroidException
from thyroid.logger import logging

class ModelResolver:
    def __init__(self, model_registery:str = "saved_models",
                transformer_dir_name="transformer",
                target_encoder_dir_name='target_encoder',
                model_dir_name='model'):
        
        self.model_registery = model_registery
        os.makedirs(self.model_registery, exist_ok=True)
        self.transformer_dir_name = transformer_dir_name 
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name

    def get_latest_dir_path(self)->Optional[str]:
        try:
            # get a list of all directories in the model registry
            dir_names = os.listdir(self.model_registery)
            
            # if the model registry is empty, return None
            if len(dir_names)==0:
                return None

            # map directory names to integers and get the maximum directory name
            dir_names = list(map(int,dir_names))
            latest_dir_name = max(dir_names)

            # return the path to the latest directory in the model registry
            return os.path.join(self.model_registery, f"{latest_dir_name}")
        
        except Exception as e:
            raise ThyroidException(e, sys)

    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)

    def get_latest_transformer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transformer is not available")
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)
    
    def get_latest_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Target encoder is not available")
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)
    
    def get_latest_save_dir_path(self)->str:
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registery,f"{0}")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registery,f"{latest_dir_num+1}")
        except Exception as e:
            raise ThyroidException(e, sys)
    
    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)
    
    def get_latest_save_transformer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)
    
    def get_latest_save_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)