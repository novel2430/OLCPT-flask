import pandas as pd
import openai
import os
import dask.dataframe as dd
import numpy as np
import tiktoken
from dotenv import load_dotenv

class OpenAIVectorizer:
    i = 0
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

    def __init__(self, df, text_col, key, output_name = 'Output/openai_embeddings.csv'):
        self.df = df
        self.text_col = text_col
        load_dotenv(key)
        self.key = os.getenv("OPENAI_API_KEY")
        #self.key = key
        self.output_name = output_name
        OpenAIVectorizer.i = 0 # resetting the counter at instantiation 
        
    @staticmethod
    def get_embedding(text):
        n = len(text)

        if OpenAIVectorizer.i%100==0:
            print(OpenAIVectorizer.i/n*100)
        OpenAIVectorizer.i+=1
        return openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
    
    def encode(self, encoding, text):
        return len(encoding.encode(text))

    def trim(self, text, max_tokens = 8100):
        enc = self.encoding.encode(text)[:max_tokens]
        trimmed_text = self.encoding.decode(enc)
        return trimmed_text 

    @staticmethod
    def simple_encode(text, key):
        load_dotenv(key)
        return OpenAIVectorizer.get_embedding(text)

    @staticmethod
    def simple_trim(text, max_tokens = 8100):
        encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
        enc = encoding.encode(text)[:max_tokens]
        trimmed_text = encoding.decode(enc)
        return trimmed_text

    @staticmethod
    def simple_extract_embedding_only(obj):
        return obj['data'][0]['embedding']

    def get_openai_embedding(self):
        openai.api_key = self.key
        trimmed_text_col = self.text_col.apply(self.trim)
        res = trimmed_text_col.progress_apply(OpenAIVectorizer.get_embedding)
        print('embeddings done')

        arr = self.extract_embedding_only(res)
        res_df = pd.DataFrame(arr)

        merged_df = self.merge(self.df, res_df)
        merged_df.to_csv(self.output_name, index=False)
        print(f'embeddings saved to {self.output_name}')
        return merged_df
    
    def extract_embedding_only(self, obj):
        return np.array([ obj[x]['data'][0]['embedding'] for x in obj.index ])

    def merge(self, df, embedding_df):
        merged_df = pd.concat([df, embedding_df], axis=1)
        merged_df = merged_df.drop('fulltext', axis=1)
        return merged_df