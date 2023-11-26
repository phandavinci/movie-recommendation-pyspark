from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
import os

print("Loading Hadoop and initializing Spark...")
os.environ['HADOOP_HOME'] = 'C:/hadoop'

# Set up Spark session
spark = SparkSession.builder \
        .appName("MovieRecommender") \
        .config("spark.hadoop.home.dir", "C:/hadoop") \
        .getOrCreate()

print("\nsuccess!!!\n")
# Load data
print("Loading the data...")
data = spark.read.csv("../ml-25m/ratings.csv", header=True, inferSchema=True)
print("\nsuccess!!!\n")

print("Dropping the data...")
data = data.dropna()
print("\nsuccess!!!\n")

print("Filtering the data...")
data = data.filter((col("rating") >= 1) & (col("rating") <= 5))
print("\nsuccess!!!\n")

print("Cleaning...")
data = data.dropDuplicates(["userId", "movieId"])
print("\nsuccess!!!\n")

# Split the data into training and test sets
print("Splitting and Getting Ready...")
(training, test) = data.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
print("\nsuccess!!!\n")

print("Training...")
model = als.fit(training)
print("\nsuccess!!!\n")

# Evaluate the model by computing the RMSE on the test data
print("Evaluating and Saving Weights...")
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) = {rmse}")

# Save the model
model.write().overwrite().save('weights')
print("\nsuccess!!!\n")
