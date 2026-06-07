import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_similarity(movies_df):
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_df['tags'])
    return cosine_similarity(tfidf_matrix)

def build_glove_similarity(movies_df, glove_model):
    def get_glove_vector(tags):
        words = tags.split()
        vectors = [glove_model[word] for word in words if word in glove_model]
        if not vectors:
            return np.zeros(100)
        return np.mean(vectors, axis=0)
    
    glove_matrix = np.vstack(movies_df['tags'].apply(get_glove_vector))
    return cosine_similarity(glove_matrix)

def get_content_recommendations(movie, movies_df, tfidf_similarity, glove_similarity, n=5, alpha=0.5):
    matches = movies_df[movies_df['title'] == movie]
    if matches.empty:
        return f"Movie '{movie}' not found"
    
    idx = matches.index[0]
    
    tfidf_scores = tfidf_similarity[idx]
    glove_scores = glove_similarity[idx]
    
    tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min())
    glove_norm = (glove_scores - glove_scores.min()) / (glove_scores.max() - glove_scores.min())
    
    combined = alpha * tfidf_norm + (1 - alpha) * glove_norm
    scores = sorted(list(enumerate(combined)), key=lambda x: x[1], reverse=True)
    
    return [(movies_df.iloc[i]['title'], round(float(score), 3)) for i, score in scores[1:n+1]]