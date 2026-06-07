import streamlit as st
import pickle
import numpy as np
import requests
import pandas as pd
import os
from huggingface_hub import hf_hub_download
from src.content_model import get_content_recommendations
from src.hybrid_model import get_hybrid_recommendations
from src.explainer import explain_recommendation, get_recommendations_with_explanation

# ── Config ──────────────────────────────────────────
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "ecb335ae831a2e50245028df40a84397")

# ── Load models ─────────────────────────────────────
@st.cache_resource
def load_models():
    movies = pickle.load(open('models/movies.pkl', 'rb'))
    movie_details = pickle.load(open('models/movie_details.pkl', 'rb'))
    links = pd.read_csv('data/raw/links.csv')

    if os.path.exists('models/tfidf_similarity.pkl'):
        tfidf_similarity = pickle.load(open('models/tfidf_similarity.pkl', 'rb'))
        glove_similarity = pickle.load(open('models/glove_similarity.pkl', 'rb'))
        svd = pickle.load(open('models/svd_model.pkl', 'rb'))
    else:
        tfidf_path = hf_hub_download(repo_id="vaibhav343/movie-recommender-models", filename="tfidf_similarity.pkl")
        glove_path = hf_hub_download(repo_id="vaibhav343/movie-recommender-models", filename="glove_similarity.pkl")
        svd_path = hf_hub_download(repo_id="vaibhav343/movie-recommender-models", filename="svd_model.pkl")
        tfidf_similarity = pickle.load(open(tfidf_path, 'rb'))
        glove_similarity = pickle.load(open(glove_path, 'rb'))
        svd = pickle.load(open(svd_path, 'rb'))

    return movies, movie_details, tfidf_similarity, glove_similarity, svd, links

movies, movie_details, tfidf_similarity, glove_similarity, svd, links = load_models()

# ── Poster fetcher ───────────────────────────────────
def get_poster(movie_id):
    tmdb_id = links[links['movieId'] == movie_id]['tmdbId'].values
    if len(tmdb_id) == 0:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id[0])}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path', '')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None
    return None

# ── UI ───────────────────────────────────────────────
st.title("🎬 Movie Recommender")
st.markdown("Get personalized movie recommendations powered by AI")

movie_list = movies['title'].values

user_id = st.number_input(
    "Enter your MovieLens User ID (optional, leave 0 to skip)",
    min_value=0, step=1, value=0, key="user_id"
)

selected_movie = st.selectbox("Search for a movie", movie_list)

if st.button("Get Recommendations"):
    with st.spinner("Finding recommendations..."):

        if user_id > 0:
            recs = get_recommendations_with_explanation(
                user_id, selected_movie, movies, svd,
                tfidf_similarity, glove_similarity, movie_details
            )
        else:
            content_recs = get_content_recommendations(
                selected_movie, movies, tfidf_similarity, glove_similarity
            )
            recs = [{'title': t, 'score': s, 'reasons': explain_recommendation(selected_movie, t, movie_details)}
                    for t, s in content_recs]

        st.subheader(f"Top 5 recommendations for {selected_movie}")
        cols = st.columns(5)

        for idx, rec in enumerate(recs):
            with cols[idx]:
                movie_id = movies[movies['title'] == rec['title']]['movieId'].values
                if len(movie_id) > 0:
                    poster = get_poster(movie_id[0])
                    if poster:
                        st.image(poster)
                    else:
                        st.image("https://via.placeholder.com/150x225?text=No+Poster")
                st.markdown(f"**{rec['title']}**")
                st.markdown(f"Score: {rec['score']}")
                for reason in rec['reasons']:
                    st.markdown(f"• {reason}")