import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Carregar os datasets
ratings = pd.read_csv("ratings_small.csv")  # Contém userId, movieId, rating
movies = pd.read_csv("movies_metadata.csv", low_memory=False)  # Pode ter colunas problemáticas

# Converter 'id' para inteiro, ignorando erros
movies = movies[pd.to_numeric(movies["id"], errors="coerce").notna()]
movies["id"] = movies["id"].astype(int)

# Mesclar os datasets
df = ratings.merge(movies, left_on="movieId", right_on="id")

# Exibir os primeiros resultados
#print(df.head())



# Separar os gêneros dos filmes
df["genres"] = df["genres"].str.split("|")

# Criar uma matriz de gêneros binária (one-hot encoding)
genres_df = df.explode("genres").pivot_table(index="movieId", columns="genres", aggfunc="size", fill_value=0)

# Exibir a matriz de gêneros
#print(genres_df.head())



# Agrupar as avaliações por usuário e calcular a média das notas por filme
user_profile = df.groupby(["userId", "movieId"])["rating"].mean().unstack()

# Exibir uma amostra do perfil do usuário
#print(user_profile.head())



# Calcular a similaridade entre os filmes com base nos gêneros
similarity_matrix = cosine_similarity(genres_df)

# Transformar em um DataFrame para facilitar a manipulação
similarity_df = pd.DataFrame(similarity_matrix, index=genres_df.index, columns=genres_df.index)

# Exibir uma amostra da matriz de similaridade
#print(similarity_df.head())


# Função para recomendar filmes semelhantes usando o modelo
# de a Filtragem Baseada em Conteúdo, utilizando a similaridade
# do cosseno entre os filmes com base nos gêneros.
def recommend_movies_content_based(movie_id, num_recommendations=5):
    """
    Retorna recomendações de filmes semelhantes com base no conteúdo (gêneros).

    Parâmetros:
    - movie_id: ID do filme de referência.
    - num_recommendations: Número de filmes recomendados.
    """

    # Buscar o título do filme correspondente ao movie_id
    movie_title = movies[movies["id"] == movie_id]["title"].values

    # Verifique se o movie_id está na matriz de similaridade
    if movie_id not in similarity_df.index:
        print(f"Erro: O movie_id {movie_id} não está na matriz de similaridade!")
        return None

    print(f"Título do filme de referência: {movie_title[0]}")

    # Encontrar os filmes mais similares ao informado
    similar_movies = similarity_df[movie_id].sort_values(ascending=False)[1:num_recommendations + 1]

    # Retornar os títulos dos filmes semelhantes
    return movies[movies["id"].isin(similar_movies.index)][["title"]]


# **Testando a Recomendação Baseada em Conteúdo**
movie_id = 3  # Escolha um filme para testar
print(recommend_movies_content_based(movie_id=movie_id, num_recommendations=5))


# Criar a matriz user-item
user_movie_ratings = df.pivot_table(index='userId', columns='movieId', values='rating')

# Substituir NaN por 0 (caso tenha filmes não avaliados)
user_movie_ratings = user_movie_ratings.fillna(0)

# Criar o modelo KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_movie_ratings.values)

# Função para recomendar filmes semelhantes usando o modelo
# de filtragem colaborativa usando o KNM
# Ajustar para garantir que o input tenha a mesma dimensão
def recommend_movies_knn(movie_id, num_recommendations=5):
    # Buscar o título do filme correspondente ao movie_id
    movie_title = movies[movies["id"] == movie_id]["title"].values

    if movie_id not in user_movie_ratings.columns:
        print(f"Erro: O movie_id {movie_id} não está na matriz de avaliações!")
        return None

    print(f"Título do filme de referência: {movie_title[0]}")

    # Encontrar o índice do filme na matriz
    movie_index = user_movie_ratings.columns.get_loc(movie_id)

    # Ajustar a entrada para ter o mesmo número de features
    movie_vector = np.zeros((1, user_movie_ratings.shape[1]))
    movie_vector[0, movie_index] = 1  # Ativar a posição do filme pesquisado

    # Encontrar os k vizinhos mais próximos
    distances, indices = knn.kneighbors(movie_vector, n_neighbors=num_recommendations + 1)

    # Pegar os filmes recomendados (excluindo o próprio filme)
    recommended_movies = user_movie_ratings.columns[indices.flatten()[1:]]

    # Retornar os títulos dos filmes recomendados
    return movies[movies["id"].isin(recommended_movies)][["title"]]


# Exemplo: recomendar 5 filmes parecidos com um filme específico usando KNN
movie_id = 3
print(recommend_movies_knn(movie_id=movie_id, num_recommendations=5))

# Geramos recomendações usando ambos os métodos:
# - KNN (Filtragem Colaborativa): Recomendação baseada nas avaliações de usuários.
# - Similaridade de Conteúdo (Cosseno): Recomendação baseada nos gêneros dos filmes.
def recommend_movies_hybrid(movie_id, num_recommendations=5, weight_knn=0.5, weight_content=0.5):
    """
    Modelo híbrido que combina Filtragem Colaborativa (KNN) e Filtragem Baseada em Conteúdo (Cosseno).

    Parâmetros:
    - movie_id: ID do filme de referência.
    - num_recommendations: Número de filmes recomendados.
    - weight_knn: Peso da recomendação KNN.
    - weight_content: Peso da recomendação baseada em conteúdo.
    """

    # Obter recomendações de ambos os métodos
    knn_recommendations = recommend_movies_knn(movie_id, num_recommendations)
    content_recommendations = recommend_movies_content_based(movie_id, num_recommendations)

    # Se algum método falhar, retorna o outro
    if knn_recommendations is None:
        return content_recommendations
    if content_recommendations is None:
        return knn_recommendations

    # Criar um dicionário para armazenar os pesos
    scores = {}

    # Adicionar pesos das recomendações KNN
    for i, row in knn_recommendations.iterrows():
        scores[row["title"]] = scores.get(row["title"], 0) + weight_knn * (num_recommendations - i)

    # Adicionar pesos das recomendações Baseadas em Conteúdo
    for i, row in content_recommendations.iterrows():
        scores[row["title"]] = scores.get(row["title"], 0) + weight_content * (num_recommendations - i)

    # Ordenar filmes por pontuação e selecionar os melhores
    sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_titles = [movie[0] for movie in sorted_movies[:num_recommendations]]

    return recommended_titles


# 🔥 **Testando a Recomendação Híbrida**
movie_id = 2  # Escolha um filme para testar
print(recommend_movies_hybrid(movie_id=movie_id, num_recommendations=5))