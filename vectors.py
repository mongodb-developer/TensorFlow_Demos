import tensorflow as tf
import numpy as np
import pymongo

# Connect to MongoDB Atlas
client = pymongo.MongoClient('mongodb+srv://user:password@clusterinfo.mongodb.net/?retryWrites=true&w=majority')
db = client['sample_mflix']
collection = db['movies']

# Prepare the data
genres = ['Western', 'History', 'Musical']
movie_data = []
labels = []

# Fetch movie data and labels from MongoDB
for movie in collection.find({}, {'vector_representation': 1, 'genres': 1}):
    if 'genres' in movie and 'vector_representation' in movie:
        genre = movie['genres'][0] if movie['genres'] else 'Unknown'
        if genre in genres:
            movie_data.append(movie['vector_representation'])
            labels.append(genres.index(genre))

# Convert the data to NumPy arrays
movie_data = np.array(movie_data)
labels = np.array(labels)

# Split the data into training and testing sets
split_index = int(0.8 * len(movie_data))
train_data = movie_data[:split_index]
train_labels = labels[:split_index]
test_data = movie_data[split_index:]
test_labels = labels[split_index:]

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(movie_data.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(genres), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

# Save the trained model
model.save('movie_genre_model.keras')
