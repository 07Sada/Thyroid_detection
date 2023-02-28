from thyroid import utils
from thyroid.entity import config_entity, artifact_entity
from thyroid.exception import ThyroidException
from thyroid.logger import logging
from typing import Optional
import os, sys
from sklearn.pipeline import Pipeline 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from thyroid.config import TARGET_COLUMN, CATEGORICAL_COLUMN
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC,RandomOverSampler,KMeansSMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

class DataTransformation:
    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig,
                        data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>'*30} Data Transformatin Initiated {'<'*30}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise ThyroidException(e, sys)
    
    @classmethod
    def get_data_transformer_object(self) -> Pipeline:
        try:
            # define pipeline with encoder
            encoder_pipeline = Pipeline(steps=[('encoder', OrdinalEncoder())])

            one_pipeline = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # define column transformer to apply encoder to specific columns
            ct = ColumnTransformer(
                transformers=[('encoder', encoder_pipeline, CATEGORICAL_COLUMN),
                ('onehot', one_pipeline, ['referral_source'])],
                remainder='passthrough'  # leave other columns unchanged
            )

            transformer = Pipeline(steps=[('transformer', ct)])
                        
            pipeline = Pipeline(steps=[('transformer', transformer)])
            
            return pipeline
        except Exception as e:
            raise ThyroidException(e, sys)

    def data_balancing(self, train_input:np.array, test_input:np.array, train_target:np.array, test_target:np.array):
        # concatenating input data
        input_data=np.vstack((train_input, test_input))
        
        # concatenating target data
        target_data=np.concatenate((train_target, test_target))

        # imputing the missing data
        imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
        input_data = imputer.fit_transform(input_data)

        # balcing the dataset
        rdsmple = RandomOverSampler()
        input_data, target_data  = rdsmple.fit_resample(input_data, target_data)

        # spliting the dataset
        X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def initiate_data_transformation(self)->artifact_entity.DataTransformationArtifact:
        try:
            # reading training and test file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)
            
            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            
            #transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            # transformation on target columns
            # target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            # target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            # transformation_pipeline = DataTransformation.get_data_transformer_object()
            # transformation_pipeline.fit(input_feature_train_df)
            
            # transforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            input_feature_train_arr, input_feature_test_arr, target_feature_train_arr, target_feature_test_arr = self.data_balancing(
                train_input = input_feature_train_arr,
                test_input = input_feature_test_arr,
                train_target = target_feature_train_arr,
                test_target =target_feature_test_arr
                )

            # rdsmple = RandomOverSampler(random_state=42)
            # logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            # input_feature_train_arr, target_feature_train_arr = rdsmple.fit_resample(input_feature_train_arr, target_feature_train_arr)
            # logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")

            # logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            # input_feature_test_arr, target_feature_test_arr = rdsmple.fit_resample(input_feature_test_arr, target_feature_test_arr)
            # logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                                    obj=transformation_pipeline)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
                                    obj=label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path)
            
            logging.info(f"Data transformation object stored")
            return data_transformation_artifact
        except Exception as e:
            raise ThyroidException(e, sys)