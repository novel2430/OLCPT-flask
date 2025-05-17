
from OpenAIVectorizer import *
from gensim.models import KeyedVectors

DATA_PATH = 'Data/'

def apply_embedding(df_text, text_col, output_name, openai_model, type='openai', key=None):

    vectorizer = OpenAIVectorizer(df_text, text_col, key=key, output_name=output_name)
    vectorizer.get_openai_embedding()
    print('OPENAI embeddings applied.')

    

