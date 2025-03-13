# System Recomendation Movies

## Movie Recommendation System Description
This system uses three approaches to recommend movies based on a user-provided movie ID (movieId):

# 1. Data Preparation

## Loading Datasets
The datasets loaded are:

ratings_small.csv: Contains user ratings of movies. 

Structure:
[ratings_small.csv](./ratings_small.csv)


movies_metadata.csv: Contains information about movies, including titles and genres. 

Structure:
[movies_metadata.csv](./movies_metadata.csv)

## Data Handling
- The id column in movies_metadata.csv is converted to an integer, ignoring invalid values.
- The data from the two datasets is merged based on the movieId field.

# 2. Creating Recommendation Models

## 2.1. Content-Based Filtering

Content-based recommendation uses movie genres to find those most similar to the reference movie.

Steps performed:

- 1 - The "genres" column of movies_metadata.csv is converted to genre lists.


- 2 - A one-hot encoding was created for the genres, generating a binary matrix (genres_df).


- 3 - The cosine similarity between the movies was calculated to build a similarity matrix (similarity_matrix).


- 4 - The recommend_movies_content_based(movie_id, num_recommendations=5) function was implemented:
  - Checks if the movie_id is in the similarity matrix.
  - Retrieves the most similar movies.
  - Returns a list with the titles, release dates, and average ratings.

## 2.2. Collaborative Filtering (KNN)

Collaborative recommendation finds similar movies based on user ratings.

### Steps performed:
- 1 - Created a user-item matrix (user_movie_ratings), where:
  - The rows represent users.

  - The columns represent movies.
  - The values are the ratings given by users to the movies.
  

- 2 - NaN values (absence of ratings) are replaced by 0.


- 3 - Created a KNN model using the cosine similarity metric.

- 4 - Implemented the function recommend_movies_knn(movie_id, num_recommendations=5):
  - Gets the position of the movie in the matrix.
  - Uses knn.kneighbors() to find the most similar movies based on the users who rated them.
  - Returns the recommended movies with title, release date and average rating.

# Features
## - Filtering Model:

- KNN (Collaborative Filtering): Recommendation based on user ratings.
- Content Similarity (Cosine): Recommendation based on movie genres.
- Hybrid Model (Combination of Both): Combines the two previous methods and orders the recommendations by the best average rating.


# Libraries

- pandas: For data manipulation and analysis.
- sklearn.metrics.pairwise.cosine_similarity: For calculating similarity between movies.
- sklearn.neighbors.NearestNeighbors: For KNN-based collaborative filtering model.
- numpy: For mathematical operations and data vectorization.

## Usage

1. Run the app:
  ```bash
  python app.py
  ```