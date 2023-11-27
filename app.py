import streamlit as st
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import desc, col

import findspark
findspark.init()

# Set up Spark session
spark = SparkSession.builder \
    .appName("MovieRecommender") \
    .config("spark.hadoop.home.dir", "C:/hadoop") \
    .config("spark.executor.heartbeatInterval", "1000000s") \
    .config("spark.network.timeout", "3000000s") \
    .getOrCreate()

# Load the saved model
model = ALSModel.load("weights")

# Load movie data
movies = spark.read.csv("../ml-25m/movies.csv", header=True, inferSchema=True)
links = spark.read.csv("../ml-25m/links.csv", header=True, inferSchema=True)
tags = spark.read.csv("../ml-25m/tags.csv", header=True, inferSchema=True)
ratings = spark.read.csv("../ml-25m/ratings.csv", header=True, inferSchema=True)

# Combine movie data with links and genres
movies = movies.join(links, on="movieId")
movies = movies.withColumn("genres", col("genres").cast("string"))  # Ensure genres are in string format

# Streamlit app
st.title("Advanced Movie Recommender System")

# Sidebar for navigation
option = st.sidebar.selectbox("Navigation", ["Recommended Movies", "All Movies"])

# User input for user ID
user_id = st.sidebar.number_input("Enter your User ID", min_value=1, max_value=138494, value=1, step=1)

# Define schema for user input
user_schema = StructType([StructField("userId", IntegerType(), True)])

# Create DataFrame for user input
user_df = spark.createDataFrame([(user_id,)], schema=user_schema)

# Generate movie recommendations for the user
user_recommendations = model.recommendForUserSubset(user_df, 100).collect()[0]['recommendations']

# Display recommendations based on the selected option
if option == "Recommended Movies":
    st.subheader(f"Top 100 Recommended Movies for User {user_id}:")

    # Assuming movies is a list of dictionaries with a 'movieId' field
    movies_dict = {row['movieId']: row for row in movies.collect()}

    sort_options = ["Title", "Genres", "Rating"]
    selected_sort_option = st.selectbox("Sort Movies By:", sort_options)

    if selected_sort_option == "Title":
        user_recommendations = sorted(user_recommendations, key=lambda x: movies_dict[x['movieId']]['title'])
    elif selected_sort_option == "Genres":
        user_recommendations = sorted(user_recommendations, key=lambda x: movies_dict[x['movieId']]['genres'])
    elif selected_sort_option == "Rating":
        user_recommendations = sorted(user_recommendations, key=lambda x: x['rating'], reverse=True)

    for i, recommendation in enumerate(user_recommendations):
        movie_info = movies_dict[recommendation['movieId']]
        st.write(f"{i+1}. Movie ID: {recommendation['movieId']}")
        st.write(f"   Title: {movie_info['title']}")
        st.write(f"   Genres: {movie_info['genres']}")
        st.write(f"   IMDb ID: {movie_info['imdbId']}")
        st.write(f"   Rating: {recommendation['rating']:.2f}")
        st.write("")

elif option == "All Movies":
    st.subheader("Top 1000 Movies:")
    sort_options_all_movies = ["Title", "Genres", "Average Rating"]
    selected_sort_option_all_movies = st.selectbox("Sort Movies By:", sort_options_all_movies)

    if selected_sort_option_all_movies == "Title":
        movies = movies.orderBy("title")
    elif selected_sort_option_all_movies == "Genres":
        movies = movies.orderBy("genres")
    elif selected_sort_option_all_movies == "Average Rating":
        top_movies = ratings.groupBy("movieId").agg({"rating": "avg"}).orderBy(desc("avg(rating)")).limit(1000)
        movies = movies.join(top_movies, on="movieId").orderBy(desc("avg(rating)"))

    for i, row in enumerate(movies.limit(1000).collect()):
        st.write(f"{i+1}. Movie ID: {row['movieId']}")
        st.write(f"   Title: {row['title']}")
        st.write(f"   Genres: {row['genres']}")
        st.write(f"   IMDb ID: {row['imdbId']}")
        st.write("")
