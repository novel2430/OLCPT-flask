# Import
from typing import Any
from pandas.io.parsers import TextFileReader
from pycaret.regression import RegressionExperiment, load_model, predict_model
import os
import pandas as pd
import numpy as np
import openai
from OpenAIVectorizer import OpenAIVectorizer
from embedding_utilities import *
from tqdm import tqdm

import embedding_utilities
tqdm.pandas(desc="Processing")
import random 
import time
import joblib
from sklearn.pipeline import Pipeline

class ScoringCSV:
    def __init__(self) -> None:
        self.__setup_global__()

    def __setup_global__(self) -> None:
        self.root_directory = '.'
        self.data_directory = 'Data'
        self.models_directory = 'Models'
        self.embed_path = 'Embeddings/'
        self.output_path = 'Predictions/'
        self.key_file = 'Keys/key.env'
        self.DATA_PATH = os.path.abspath(os.path.join(self.root_directory, self.data_directory))
        self.input_data = 'text4scoring_csv-test'
        self.model_names_openai = ['openai_extra', 'openai_openn', 'openai_consc', 'openai_agree', 'openai_neuro','openai_narc','openai_humility']


    def __build_joblib_cache_dir__(self):
        self.cache_dir = os.path.abspath('./.joblib_cache/csv_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = self.cache_dir

    def __load_dataset_(self) -> pd.DataFrame | pd.Series | Any:
        try:
            df = pd.read_csv(os.path.join(self.DATA_PATH, self.input_data+'.csv'), encoding='utf-8')
            print('Read the data usnig utf-8 encoding')
        except:
            df = pd.read_csv(os.path.join(self.DATA_PATH, self.input_data+'.csv'), encoding='ISO-8859-1')
            print('Read the data using ISO-8859-1 encoding')
        return df

    def __load_custom_dataset_(self, input_file:str) -> pd.DataFrame | pd.Series | Any:
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
            print('Read the data usnig utf-8 encoding')
        except:
            df = pd.read_csv(input_file, encoding='ISO-8859-1')
            print('Read the data using ISO-8859-1 encoding')
        return df

    def __load_openai_model__(self) -> list[Any | None]:
        models = [load_model(f'{self.models_directory}/{model_name}_model') for model_name in self.model_names_openai]
        # Slove pycaret and joblib cache error
        for m in models:
            if isinstance(m, Pipeline):
                m.memory = joblib.Memory(location=self.cache_dir)
        return models

    def __write_openai_gpt_embedding_file__(self, df: pd.DataFrame | pd.Series ) -> pd.DataFrame | None:
        # Define the range of observations to score
        start_row = 0
        end_row = df.shape[0]
        text_col = df['fulltext']

        # Define file name for sample embeddings
        embedding_file_name = self.embed_path + self.input_data + '_openai_embeddings_' + str(start_row) + '-' + str(end_row) + '.csv'

        # Initialize a flag to keep track of successful embeddings
        all_rows_embedded = False

        # Get the OpenAI embeddings for the data  [This may take a minute]
        while not all_rows_embedded and text_col is not None:
            try:
                embeddings = apply_embedding(df[start_row:end_row], text_col[start_row:end_row], embedding_file_name, None, type='openai', key=self.key_file)
                all_rows_embedded = True  # Set flag to True if embeddings are successful for all rows
            except openai.OpenAIError as e:
                randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                sleep_dur = 20 ** start_row + randomness_collision_avoidance  # Exponential backoff based on start_row
                print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                time.sleep(sleep_dur)

        if all_rows_embedded:
            return pd.read_csv(embedding_file_name)

    def predict_attribute_openai(self, enc_df_oai, models, df):
        predictions = [predict_model(model, data=enc_df_oai)[['prediction_label']] for model in models]
        # rename the predictions output to the attribute names
        for i in range(len(self.model_names_openai)):
            predictions[i] = predictions[i].rename(columns={'prediction_label': self.model_names_openai[i]})
        # concatenate the predictions
        predictions = pd.concat(predictions, axis=1)
        output_df_oai = pd.concat([df, predictions], axis=1)
        output_df_oai = output_df_oai.drop(['fulltext'], axis=1)
        return output_df_oai

    def run(self, input_path:str, output_csv_path:str):
        print("######## Setup Cache Dir ########")
        self.__build_joblib_cache_dir__()
        # load dataset
        print("######## Load Dataset ########")
        # df = self.__load_dataset_()
        df = self.__load_custom_dataset_(input_path)
        # Load Trained OpenAI GPT Models
        print("######## Load Openai Model ########")
        models = self.__load_openai_model__()
        # Apply OpenAI GPT Embeddings
        print("######## Do Openai Embeddings ########")
        enc_df_oai = self.__write_openai_gpt_embedding_file__(df)

        if enc_df_oai is None:
            print("ERROR :(")
        else:
            print("######## Do Prediction ########")
            predict = self.predict_attribute_openai(enc_df_oai, models, df)
            print(predict)
            print("######## Write File to {} ########".format(output_csv_path))
            predict.to_csv(output_csv_path, index=False) 

# scoring = ScoringCSV()
# scoring.run()
