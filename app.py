import streamlit as st
import pickle
import numpy as np
import requests
from src.content_model import get_content_recommendations
from src.hybrid_model import get_hybrid_recommendations
from src.explainer import get_recommendations_with_explanation

# Load all models
@st.cache_resource
def load_models():
    movies = pickle.load(open('models/movies.pkl', 'rb'))
    movie_details = pickle.load(open('models/movie_details.pkl', 'rb'))
    tfidf_similarity = pickle.load(open('models/tfidf_similarity.pkl', 'rb'))
    glove_similarity = pickle.load(open('models/glove_similarity.pkl', 'rb'))
    svd = pickle.load(open('models/svd_model.pkl', 'rb'))
    return movies, movie_details, tfidf_similarity, glove_similarity, svd

movies, movie_details, tfidf_similarity, glove_similarity, svd = load_models()