import streamlit as st
import pickle
import numpy as np
import requests
import pandas as pd
import os
from huggingface_hub import hf_hub_download
from src.content_model import get_content_recommendations
from src.explainer import explain_recommendation

# ── Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="CineAI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #141414;
    color: #ffffff;
}

header, footer { visibility: hidden; }
.block-container { padding: 0 2rem; max-width: 1400px; }

/* Hero */
.hero {
    background: linear-gradient(180deg, #1a1a2e 0%, #141414 100%);
    padding: 4rem 2rem 3rem 2rem;
    margin: -1rem -2rem 2rem -2rem;
    text-align: center;
    border-bottom: 1px solid #222;
}

.hero-logo {
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 0.3rem;
    color: #e50914;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}

.hero-title span {
    color: #e50914;
}

.hero-sub {
    font-size: 1.1rem;
    color: #888;
    font-weight: 400;
    margin-bottom: 0;
}

/* Search */
.stSelectbox > div > div {
    background-color: #1f1f1f !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1rem !important;
}

.stSelectbox label {
    color: #888 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05rem !important;
}

/* Button */
.stButton > button {
    background: #e50914 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.85rem 2.5rem !important;
    width: 100% !important;
    letter-spacing: 0.02rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: #f40612 !important;
    transform: scale(1.02) !important;
}

/* Section header */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 2.5rem 0 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #222;
    margin-left: 1rem;
}

/* Movie card */
.movie-card {
    background: #1f1f1f;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2a2a2a;
    transition: all 0.2s ease;
    height: 100%;
}

.movie-card:hover {
    border-color: #e50914;
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

.movie-poster-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    background: #2a2a2a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
}

.movie-body {
    padding: 1rem;
}

.movie-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.4rem;
    line-height: 1.3;
}

.movie-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
}

.badge-rating {
    background: #f5c518;
    color: #000;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
}

.badge-match {
    background: #e50914;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
}

.movie-overview {
    font-size: 0.78rem;
    color: #999;
    line-height: 1.5;
    margin-bottom: 0.75rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.reasons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
}

.reason-chip {
    background: #2a2a2a;
    color: #aaa;
    font-size: 0.68rem;
    padding: 0.2rem 0.5rem;
    border-radius: 20px;
    border: 1px solid #333;
    white-space: nowrap;
}

/* Spinner */
div[data-testid="stSpinner"] {
    color: #e50914 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Config ───────────────────────────────────────────
TMDB_API_KEY = os.environ["TMDB_API_K

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

# ── TMDB ─────────────────────────────────────────────
@st.cache_data
def get_movie_details_tmdb(movie_id):
    tmdb_id = links[links['movieId'] == movie_id]['tmdbId'].values
    if len(tmdb_id) == 0:
        return None, None, None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id[0])}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        poster = f"https://image.tmdb.org/t/p/w500{data['poster_path']}" if data.get('poster_path') else None
        overview = data.get('overview', '')
        rating = data.get('vote_average', None)
        return poster, overview, round(rating, 1) if rating else None
    except:
        return None, None, None

# ── Hero ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">✦ CineAI</div>
    <div class="hero-title">Find your next<br><span>favorite film</span></div>
    <div class="hero-sub">AI-powered recommendations based on what you love</div>
</div>
""", unsafe_allow_html=True)

# ── Search ───────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    movie_list = sorted(movies['title'].values)
    selected_movie = st.selectbox("Choose a movie you love", movie_list)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    recommend_btn = st.button("✦ Recommend")

# ── Results ──────────────────────────────────────────
if recommend_btn:
    with st.spinner("Finding your perfect movies..."):
        content_recs = get_content_recommendations(
            selected_movie, movies, tfidf_similarity, glove_similarity
        )
        recs = [{'title': t, 'score': s,
                 'reasons': explain_recommendation(selected_movie, t, movie_details)}
                for t, s in content_recs]

    st.markdown(f'<div class="section-header">Because you liked <em>{selected_movie}</em></div>',
                unsafe_allow_html=True)

    cols = st.columns(5)
    for idx, rec in enumerate(recs):
        with cols[idx]:
            movie_id = movies[movies['title'] == rec['title']]['movieId'].values
            poster, overview, rating = None, '', None
            if len(movie_id) > 0:
                poster, overview, rating = get_movie_details_tmdb(int(movie_id[0]))

            card_html = f"""
            <div class="movie-card">
            """
            st.markdown(card_html, unsafe_allow_html=True)

            if poster:
                st.image(poster, use_column_width=True)
            else:
                st.markdown('<div class="movie-poster-placeholder">🎬</div>', unsafe_allow_html=True)

            match_pct = round(rec['score'] * 100)
            rating_badge = f'<span class="badge-rating">⭐ {rating}</span>' if rating else ''
            reasons_html = ''.join([f'<span class="reason-chip">{r}</span>' for r in rec['reasons']])
            overview_text = overview[:150] + '...' if overview else ''

            st.markdown(f"""
            <div class="movie-body">
                <div class="movie-title">{rec['title']}</div>
                <div class="movie-meta">
                    {rating_badge}
                    <span class="badge-match">{match_pct}% match</span>
                </div>
                <div class="movie-overview">{overview_text}</div>
                <div class="reasons">{reasons_html}</div>
            </div>
            </div>
            """, unsafe_allow_html=True)