from dataclasses import dataclass
import pymongo
from datetime import datetime 
import os, sys 
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EnvironmentVariable():
        mongo_db_url = os.getenv("MONGO_DB_URL")

env_var = EnvironmentVariable()

mongo_client = pymongo.MongoClient(env_var.mongo_db_url)

TARGET_COLUMN = "Class"

NUMERICAL_COLUMN = ['age', 'T3', 'TT4', 'T4U', 'FTI']

CATEGORICAL_COLUMN = ['sex','on_thyroxine', 'query_on_thyroxine','on_antithyroid_medication','sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid','query_hyperthyroid',
                        'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'referral_source']