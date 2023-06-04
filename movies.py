import tensorflow as tf
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
for movie in collection.find({}, {'plot': 1, 'genres': 1}):
    if 'genres' in movie and 'plot' in movie:
        genre = movie['genres'][0] if movie['genres'] else 'Unknown'
        if genre in genres:
            movie_data.append(movie['plot'])
            labels.append(genres.index(genre))

# Tokenize the movie data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(movie_data)
sequences = tokenizer.texts_to_sequences(movie_data)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Pad sequences to have the same length
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

# Convert to TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))
dataset = dataset.shuffle(len(dataset)).batch(32)

# Split the data into training and testing sets
split_index = int(0.8 * len(dataset))
train_data = dataset.take(split_index)
test_data = dataset.skip(split_index)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_sequence_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(genres), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, validation_data=test_data)

# Save the trained model
model.save('movie_genre_model.keras')
