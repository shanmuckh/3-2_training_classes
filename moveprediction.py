import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'title': ['Avatar', 'Titanic', 'Avengers', 'Iron Man', 'The Notebook'],
    'genres': [
        'Action Adventure Fantasy',
        'Romance Drama',
        'Action Adventure Sci-Fi',
        'Action Sci-Fi',
        'Romance Drama'
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['genres'])

similarity = cosine_similarity(tfidf_matrix)

def recommend_movie(movie_name, top_n=3):
    if movie_name not in df['title'].values:
        print("Movie not found")
        return

    index = df[df['title'] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"Recommended movies for '{movie_name}':")
    for i in scores[1:top_n+1]:
        print(df.iloc[i[0]]['title'])

recommend_movie("Avatar")
