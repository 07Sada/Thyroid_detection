import pymongo 
import pandas as pd 
import json 
from thyroid.config import mongo_client

DATA_FILE_PATH = "/config/workspace/hypothyroid_cleaned.csv"
DATABASE_NAME = 'thyroid-deases'
COLLECTION_NAME = 'thyroid'

if __name__ =="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Data loaded Successfully")

    # converting dataframe to json so that we can dump the records in mongo db
    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    #df.T.json() transposing the dataframe and converting to json format
    #json.loads() function converts the json string to python dictonary
    #values method of dictonary is called which return a list of all the values in the dictonary
    #list function is used to convert the returned values to list

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
    print("Data Dumped Successfully")

