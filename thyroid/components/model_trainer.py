from thyroid.entity import config_entity, artifact_entity
from thyroid.exception import ThyroidException
from thyroid.logger import logging
from typing import Optional, List
from thyroid import utils
from sklearn.metrics import f1_score 
from xgboost import XGBClassifier
import os, sys

class ModelTrainer:
    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                        data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>'*30} Model Training Initiated {'<'*30}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        
        except Exception as e:
            raise ThyroidException(e, sys)

    def train_model(self, x, y):
        try:
            xgb_clf = XGBClassifier(objective='multi:softmax', 
                            num_class=4,  
                            early_stopping_rounds=15, 
                            eval_metric=['merror','mlogloss'], 
                            seed=42)
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting the input and target features from both train and test set")
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:, :-1], test_arr[:,-1]

            logging.info(f"Train the model")
            model = self.train_model(x=x_train, y=y_train)

            logging.info(f"Calculating the f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train, average='weighted')

            logging.info(f"Calculating the f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test, average='weighted')

            logging.info(f"train score: {f1_train_score} and test score {f1_test_score}")
            # check for overfitting or underfitting 
            logging.info(f"Checking if our model is underfitting or not")
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give expected accuracy: [{self.model_trainer_config.expected_score}]\
                    model actual score: [{f1_test_score}]")

            logging.info(f"checking if our model is overfitting or not")
            diff = abs(f1_train_score - f1_test_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff:[{diff}] is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")
            
            # save the trained model
            logging.info(f"Saving the model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # preparing the artifact
            logging.info(f"Preparing the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
                                                                            f1_train_score=f1_train_score, 
                                                                            f1_test_score=f1_test_score)
            logging.info(f"Model Trainer Artifact Stored")
            return model_trainer_artifact
        except Exception as e:
            raise ThyroidException(e, sys)
