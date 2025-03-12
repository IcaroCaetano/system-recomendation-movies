import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Carregar os datasets
ratings = pd.read_csv("ratings_small.csv")  # Contém userId, movieId, rating
movies = pd.read_csv("movies_metadata.csv", low_memory=False)  # Pode ter colunas problemáticas

# Converter 'id' para inteiro, ignorando erros
movies = movies[pd.to_numeric(movies["id"], errors="coerce").notna()]
movies["id"] = movies["id"].astype(int)

# Mesclar os datasets
df = ratings.merge(movies, left_on="movieId", right_on="id")

# Exibir os primeiros resultados
print(df.head())

print("*************************************************************")

# Separar os gêneros dos filmes
df["genres"] = df["genres"].str.split("|")

# Criar uma matriz de gêneros binária (one-hot encoding)
genres_df = df.explode("genres").pivot_table(index="movieId", columns="genres", aggfunc="size", fill_value=0)

# Exibir a matriz de gêneros
print(genres_df.head())

print("*************************************************************")

# Agrupar as avaliações por usuário e calcular a média das notas por filme
user_profile = df.groupby(["userId", "movieId"])["rating"].mean().unstack()

# Exibir uma amostra do perfil do usuário
print(user_profile.head())

print("*************************************************************")

# Calcular a similaridade entre os filmes com base nos gêneros
similarity_matrix = cosine_similarity(genres_df)

# Transformar em um DataFrame para facilitar a manipulação
similarity_df = pd.DataFrame(similarity_matrix, index=genres_df.index, columns=genres_df.index)

# Exibir uma amostra da matriz de similaridade
print(similarity_df.head())

# Função para recomendar filmes semelhantes
def recommend_movies(movie_id, num_recommendations=5):
    # Verifique se o movie_id está na matriz de similaridade
    if movie_id not in similarity_df.index:
        print(f"Erro: O movie_id {movie_id} não está na matriz de similaridade!")
        return None

    # Encontre filmes semelhantes
    similar_movies = similarity_df[movie_id].sort_values(ascending=False)[1:num_recommendations+1]

    # Retorne os títulos dos filmes semelhantes
    return movies[movies["id"].isin(similar_movies.index)][["title"]]

# Exemplo: recomendar 5 filmes parecidos com o filme com movie_id=1
movie_id = 2
print(recommend_movies(movie_id=movie_id, num_recommendations=5))

