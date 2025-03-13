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


# Criar um dataframe com a média das avaliações dos filmes
average_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
average_ratings.rename(columns={"rating": "avg_rating"}, inplace=True)

# Juntar essa informação com o dataset de filmes
movies_with_ratings = movies.merge(average_ratings, left_on="id", right_on="movieId", how="left").fillna(0)




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
    Retorna recomendações de filmes semelhantes com base no conteúdo (gêneros),
    listando título, data de lançamento e média de avaliações.
    """
    movie_title = movies[movies["id"] == movie_id]["title"].values

    if movie_id not in similarity_df.index:
        print(f"Erro: O movie_id {movie_id} não está na matriz de similaridade!")
        return None

    print(f"Título do filme de referência: {movie_title[0]}")

    similar_movies = similarity_df[movie_id].sort_values(ascending=False)[1:num_recommendations + 1]

    return movies_with_ratings[movies_with_ratings["id"].isin(similar_movies.index)][["title", "release_date", "avg_rating"]]



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
    """
    Retorna recomendações usando KNN (Filtragem Colaborativa),
    listando título, data de lançamento e média de avaliações.
    """

    movie_title = movies[movies["id"] == movie_id]["title"].values

    if movie_id not in user_movie_ratings.columns:
        print(f"Erro: O movie_id {movie_id} não está na matriz de avaliações!")
        return None

    print(f"Título do filme de referência: {movie_title[0]}")

    movie_index = user_movie_ratings.columns.get_loc(movie_id)

    movie_vector = np.zeros((1, user_movie_ratings.shape[1]))
    movie_vector[0, movie_index] = 1

    distances, indices = knn.kneighbors(movie_vector, n_neighbors=num_recommendations + 1)

    recommended_movies = user_movie_ratings.columns[indices.flatten()[1:]]

    return movies_with_ratings[movies_with_ratings["id"].isin(recommended_movies)][["title", "release_date", "avg_rating"]]





# Geramos recomendações usando ambos os métodos:
# - KNN (Filtragem Colaborativa): Recomendação baseada nas avaliações de usuários.
# - Similaridade de Conteúdo (Cosseno): Recomendação baseada nos gêneros dos filmes.
def recommend_movies_hybrid(movie_id, num_recommendations=5):
    """
    Modelo híbrido que combina recomendações de conteúdo e colaborativa,
    ordenando os filmes pela melhor média de avaliação.
    """

    content_recommendations = recommend_movies_content_based(movie_id, num_recommendations * 2)
    knn_recommendations = recommend_movies_knn(movie_id, num_recommendations * 2)

    if content_recommendations is None or knn_recommendations is None:
        return None

    hybrid_recommendations = pd.concat([content_recommendations, knn_recommendations]).drop_duplicates()

    # Ordenar pela maior média de avaliações
    hybrid_recommendations = hybrid_recommendations.sort_values(by="avg_rating", ascending=False)

    return hybrid_recommendations.head(num_recommendations)


def main():
    while True:
        try:
            movie_id = int(input("\nDigite o ID do filme para recomendação (ou 0 para sair): "))

            if movie_id == 0:
                print("Saindo do sistema...")
                break

            print("\n🔹 Recomendação Baseada em Conteúdo:")
            print(recommend_movies_content_based(movie_id, num_recommendations=5))

            print("\n🔹 Recomendação Colaborativa (KNN):")
            print(recommend_movies_knn(movie_id, num_recommendations=5))

            print("\n🔹 Recomendação Híbrida:")
            print(recommend_movies_hybrid(movie_id, num_recommendations=5))

        except ValueError:
            print("❌ Erro: Digite um número válido para o movie_id.")

if __name__ == "__main__":
    main()