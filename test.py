# This will test the functionality of your TF installation does simple things
import tensorflow as tf
import pymongo

# MongoDB connection string
connection_string = 'mongodb+srv://username:password@clusterinfo.mongodb.net/sample_mflix'

# Connect to MongoDB
client = pymongo.MongoClient(connection_string)

# Access the movies collection
db = client.sample_mflix
movies_collection = db.movies

# Fetch a movie document
movie = movies_collection.find_one()

# Print the movie title
print("Movie Title:", movie['title'])

# Perform a simple TensorFlow operation
a = tf.constant(5)
b = tf.constant(10)
c = tf.add(a, b)

# Print the result
print("Result:", c.numpy())
