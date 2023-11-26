import streamlit as st
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import desc, col
import findspark
findspark.init()

# Set up Spark session
spark = SparkSession.builder.master("local") \
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
st.title("Movie Recommender System")

# User input for user ID
user_id = st.number_input("Enter your User ID", min_value=1, max_value=138494, value=1, step=1)

# Define schema for user input
user_schema = StructType([StructField("userId", IntegerType(), True)])

# Create DataFrame for user input
user_df = spark.createDataFrame([(user_id,)], schema=user_schema)

# Generate movie recommendations for the user
user_recommendations = model.recommendForUserSubset(user_df, 10).collect()[0]['recommendations']

# Display recommendations
st.subheader(f"Top 10 Movie Recommendations for User {user_id}:")
for i, recommendation in enumerate(user_recommendations):
    movie_info = movies.filter(col("movieId") == recommendation['movieId']).select("title", "genres", "imdbId").collect()[0]
    st.write(f"{i+1}. Movie ID: {recommendation['movieId']}")
    st.write(f"   Title: {movie_info['title']}")
    st.write(f"   Genres: {movie_info['genres']}")
    st.write(f"   IMDb ID: {movie_info['imdbId']}")
    st.write("")

# Display top-rated movies
st.subheader("Top-Rated Movies:")
top_movies = ratings.groupBy("movieId").agg({"rating": "avg"}).orderBy(desc("avg(rating)")).limit(10)
for i, row in enumerate(top_movies.collect()):
    movie_info = movies.filter(col("movieId") == row['movieId']).select("title", "genres", "imdbId").collect()[0]
    st.write(f"{i+1}. Movie ID: {row['movieId']}")
    st.write(f"   Title: {movie_info['title']}")
    st.write(f"   Genres: {movie_info['genres']}")
    st.write(f"   IMDb ID: {movie_info['imdbId']}")
    st.write(f"   Average Rating: {row['avg(rating)']:.2f}")
    st.write("")

# Add sorting options
sort_options = ["Title", "Genres", "Average Rating"]
selected_sort_option = st.selectbox("Sort Movies By:", sort_options)

if selected_sort_option == "Title":
    movies = movies.orderBy("title")
elif selected_sort_option == "Genres":
    movies = movies.orderBy("genres")
elif selected_sort_option == "Average Rating":
    top_movies = ratings.groupBy("movieId").agg({"rating": "avg"}).orderBy(desc("avg(rating)")).limit(10)
    movies = movies.join(top_movies, on="movieId").orderBy(desc("avg(rating)"))

# Display sorted movies
st.subheader(f"Sorted Movies by {selected_sort_option}:")
for i, row in enumerate(movies.limit(10).collect()):
    st.write(f"{i+1}. Movie ID: {row['movieId']}")
    st.write(f"   Title: {row['title']}")
    st.write(f"   Genres: {row['genres']}")
    st.write(f"   IMDb ID: {row['imdbId']}")
    st.write("")
