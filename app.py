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

# ── Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="CineAI",
    page_icon="🎬",
    layout="wide"
)

# ── Dark Cinematic CSS ───────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background-color: #0f0f0f;
        color: #ffffff;
    }
    
    /* Hide default header */
    header {visibility: hidden;}
    
    /* Title */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #e50914, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Search bar */
    .stTextInput input {
        background-color: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }

    /* Number input */
    .stNumberInput input {
        background-color: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(90deg, #e50914, #ff6b6b) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    .stButton button:hover {
        opacity: 0.85 !important;
    }

    /* Movie card */
    .movie-card {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
        transition: transform 0.2s;
        border: 1px solid #222;
        height: 100%;
    }

    .movie-card:hover {
        transform: translateY(-4px);
        border-color: #e50914;
    }

    .movie-info {
        padding: 0.75rem;
    }

    .movie-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
        line-height: 1.3;
    }

    .movie-score {
        color: #e50914;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .movie-overview {
        font-size: 0.78rem;
        color: #aaaaaa;
        line-height: 1.4;
        margin-bottom: 0.5rem;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .reason-tag {
        display: inline-block;
        background-color: #2a2a2a;
        color: #cccccc;
        font-size: 0.72rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        border: 1px solid #333;
    }

    .rating-badge {
        display: inline-block;
        background-color: #f5c518;
        color: #000000;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        border-left: 4px solid #e50914;
        padding-left: 0.75rem;
    }

    /* Selectbox */
    .stSelectbox div {
        background-color: #1a1a1a !important;
        color: white !important;
    }

    /* Spinner */
    .stSpinner {
        color: #e50914 !important;
    }

    /* Divider */
    hr {
        border-color: #222 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "ecb335ae831a2e50245028df40a84397")

# ── Load Models ──────────────────────────────────────
@st.cache_resource
def load_models():
    links = pd.read_csv('data/raw/links.csv')

    if os.path.exists('models/tfidf_similarity.pkl'):
        movies = pickle.load(open('models/movies.pkl', 'rb'))
        movie_details = pickle.load(open('models/movie_details.pkl', 'rb'))
        tfidf_similarity = pickle.load(open('models/tfidf_similarity.pkl', 'rb'))
        glove_similarity = pickle.load(open('models/glove_similarity.pkl', 'rb'))
        svd = pickle.load(open('models/svd_model.pkl', 'rb'))
    else:
        movies_path = hf_hub_download(repo_id="vaibhav343/movie-recommender-models", filename="movies.pkl")
        details_path = hf_hub_download(repo_id="vaibhav343/movie-recommender-models", filename="movie_details.pkl")
        tfidf_path = hf_hub_download(repo_id="vaibhav343/movie-recommender-models", filename="tfidf_similarity.pkl")
        glove_path = hf_hub_download(repo_id="vaibhav343/movie-recommender-models", filename="glove_similarity.pkl")
        svd_path = hf_hub_download(repo_id="vaibhav343/movie-recommender-models", filename="svd_model.pkl")
        movies = pickle.load(open(movies_path, 'rb'))
        movie_details = pickle.load(open(details_path, 'rb'))
        tfidf_similarity = pickle.load(open(tfidf_path, 'rb'))
        glove_similarity = pickle.load(open(glove_path, 'rb'))
        svd = pickle.load(open(svd_path, 'rb'))

    return movies, movie_details, tfidf_similarity, glove_similarity, svd, links

movies, movie_details, tfidf_similarity, glove_similarity, svd, links = load_models()

# ── TMDB Fetcher ─────────────────────────────────────
def get_movie_details_tmdb(movie_id):
    tmdb_id = links[links['movieId'] == movie_id]['tmdbId'].values
    if len(tmdb_id) == 0:
        return None, None, None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id[0])}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        poster = f"https://image.tmdb.org/t/p/w500{data.get('poster_path', '')}" if data.get('poster_path') else None
        overview = data.get('overview', '')
        rating = data.get('vote_average', None)
        return poster, overview, rating
    except:
        return None, None, None

# ── Hero ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 CineAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Personalized movie recommendations powered by AI</div>', unsafe_allow_html=True)

# ── Inputs ───────────────────────────────────────────
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    movie_list = sorted(movies['title'].values)
    selected_movie = st.selectbox("🔍 Search for a movie", movie_list)

with col2:
    user_id = st.number_input("👤 User ID (optional)", min_value=0, step=1, value=0, key="user_id")

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_btn = st.button("Get Recommendations")

# ── Results ──────────────────────────────────────────
if recommend_btn:
    with st.spinner("Finding your perfect movies..."):

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

        st.markdown(f'<div class="section-title">Top picks for "{selected_movie}"</div>', unsafe_allow_html=True)

        cols = st.columns(5)
        for idx, rec in enumerate(recs):
            with cols[idx]:
                movie_id = movies[movies['title'] == rec['title']]['movieId'].values
                poster, overview, rating = None, '', None
                if len(movie_id) > 0:
                    poster, overview, rating = get_movie_details_tmdb(movie_id[0])

                # Poster
                if poster:
                    st.image(poster, use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450/1a1a1a/666666?text=No+Poster", use_column_width=True)

                # Info
                st.markdown(f'<div class="movie-title">{rec["title"]}</div>', unsafe_allow_html=True)

                if rating:
                    st.markdown(f'<span class="rating-badge">⭐ {round(rating, 1)}</span>', unsafe_allow_html=True)

                st.markdown(f'<div class="movie-score">Match: {round(rec["score"] * 100)}%</div>', unsafe_allow_html=True)

                if overview:
                    st.markdown(f'<div class="movie-overview">{overview[:150]}...</div>', unsafe_allow_html=True)

                reasons_html = ''.join([f'<span class="reason-tag">{r}</span>' for r in rec['reasons']])
                st.markdown(f'<div>{reasons_html}</div>', unsafe_allow_html=True)