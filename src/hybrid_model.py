def get_hybrid_recommendations(user_id, movie, tmdb_ml, svd, tfidf_similarity, glove_similarity, n=5, alpha=0.5):
    content_recs = get_content_recommendations(movie, tmdb_ml, tfidf_similarity, glove_similarity, n=20)
    
    hybrid_scores = []
    for title, content_score in content_recs:
        movie_row = tmdb_ml[tmdb_ml['title'] == title]
        if movie_row.empty:
            continue
        movie_id = movie_row['movieId'].values[0]
        svd_score = svd.predict(user_id, movie_id).est
        final_score = alpha * content_score + (1 - alpha) * (svd_score / 5.0)
        hybrid_scores.append((title, round(final_score, 3)))
    
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    return hybrid_scores[:n]