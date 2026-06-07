from src.hybrid_model import get_hybrid_recommendations
def explain_recommendation(movie1, movie2, movie_details):
    m1 = movie_details[movie_details['title'] == movie1].iloc[0]
    m2 = movie_details[movie_details['title'] == movie2].iloc[0]
    
    reasons = []
    
    common_genres = set(m1['genres']) & set(m2['genres'])
    if common_genres:
        reasons.append(f"Similar genres: {', '.join(common_genres)}")
    
    common_cast = set(m1['cast']) & set(m2['cast'])
    if common_cast:
        reasons.append(f"Shared actors: {', '.join(common_cast)}")
    
    common_crew = set(m1['crew']) & set(m2['crew'])
    if common_crew:
        reasons.append(f"Same director: {', '.join(common_crew)}")
    
    if not reasons:
        reasons.append("Similar themes and style")
    
    return reasons


def get_recommendations_with_explanation(user_id, movie, tmdb_ml, svd, tfidf_similarity, glove_similarity, movie_details, n=5):
    recs = get_hybrid_recommendations(user_id, movie, tmdb_ml, svd, tfidf_similarity, glove_similarity, n=n)
    
    results = []
    for title, score in recs:
        reasons = explain_recommendation(movie, title, movie_details)
        results.append({
            'title': title,
            'score': score,
            'reasons': reasons
        })
    
    return results