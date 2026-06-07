import ast
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_names(text):
    return [i['name'] for i in ast.literal_eval(text)][:5]

def extract_director(text):
    return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

def preprocess(text):
    words = [i for i in text.split() if i not in stop_words]
    lst = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lst)