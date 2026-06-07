import numpy as np
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def build_svd_model(rated_movies):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(rated_movies[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    svd = SVD(n_factors=100, random_state=42)
    svd.fit(trainset)
    
    predictions = svd.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")
    
    return svd, trainset

def get_svd_recommendations(user_id, svd, trainset, rated_movies, tmdb_ml, n=5):
    rated = rated_movies[rated_movies['userId'] == user_id]['movieId'].values
    unrated = tmdb_ml[~tmdb_ml['movieId'].isin(rated)]
    predictions = [(movie_id, svd.predict(user_id, movie_id).est) for movie_id in unrated['movieId'].values]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    results = []
    for movie_id, score in predictions[:n]:
        title = tmdb_ml[tmdb_ml['movieId'] == movie_id]['title'].values[0]
        results.append((title, round(score, 3)))
    
    return results