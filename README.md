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

## 2.3. Content-Based Filtering (NLP)
Natural Language Processing (NLP)-based recommendation finds similar movies based on textual descriptions.

### Steps performed:
- 1 - A TF-IDF embedding matrix was created from the movie descriptions (overview).
  - The descriptions were pre-processed and stopwords were removed.
  - A TF-IDF Vectorizer was applied to represent each movie as a numeric vector.


- 2 - The cosine similarity between the TF-IDF vectors was calculated to measure the proximity between the movies.


- 3 - A similarity matrix (description_similarity_df) was created, where:
  - The rows and columns represent the movies.
  - The values indicate the degree of similarity between the movies based on the description.


- 4 - Implementation of the function recommend_movies_nlp(movie_id, num_recommendations=5):
  - Checks if the movie_id is present in the similarity matrix.
  - Gets the most similar movies by sorting the values the array (description_similarity_df).
  - Returns recommended movies with title and release date.

## 2.4. Hybrid Model
This model combines the previous methods and prioritizes the movies with the best average ratings.

### Steps performed:
- 1 - Calls both recommendation functions (recommend_movies_content_based and recommend_movies_knn).


- 2 - Combines the results by removing duplicates.


- 3 - Orders the recommended movies by the average rating (avg_rating).


- 4 - Implemented the function recommend_movies_hybrid(movie_id, num_recommendations=5).

# Features
## - Filtering Model:

- KNN (Collaborative Filtering): Recommendation based on user ratings.
- Content Similarity (Cosine): Recommendation based on movie genres.
- Hybrid Model (Combination of Both): Combines the two previous methods and orders the recommendations by the best average rating.

# User Interface (CLI)
The program displays an interactive menu in the terminal, where the user can enter a movieId and view the recommendations.

## Execution flow:

- 1 - The user enters the movieId of the desired movie.


- 2 - User informs the limit of the recommendation list
 

- 3 - Recommendations are displayed for:
    - Content-Based Filtering
    - Collaborative Filtering (KNN)
    - Description-based filtering with NLP
    - Hybrid Model
    - If the user enters 0, the program ends..

# Libraries

- pandas: For data manipulation and analysis.


- sklearn.metrics.pairwise.cosine_similarity: For calculating similarity between movies.


- sklearn.neighbors.NearestNeighbors: For KNN-based collaborative filtering model.


- sklearn.feature_extraction.text: is used to convert a set of text documents into a numerical matrix of TF-IDF (Term Frequency-Inverse Document Frequency) weights.


- numpy: For mathematical operations and data vectorization.

## Usage

1. Install
  ```bash
    pip install -r requirements.txt
  ```

2. Run the app:
  ```bash
    python app.py
  ```