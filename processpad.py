import tensorflow as tf
import pymongo
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# MongoDB connection string
connection_string = 'mongodb+srv://jschmitz:slb2021@cisco-demo.tnhx6.mongodb.net/?retryWrites=true&w=majoritymongodb.net/sample_mflix'

# Connect to MongoDB
client = pymongo.MongoClient(connection_string)

# Access the movies collection
db = client.sample_mflix
movies_collection = db.movies

# Retrieve movie data from MongoDB
movies = movies_collection.find({}, {'title': 1, 'plot': 1, 'genres': 1})

# Preprocess the data
texts = []
labels = []
for movie in movies:
    title = movie.get('title', '')
    plot = movie.get('plot', '')
    genres = movie.get('genres', [])
    if title and plot and genres:
        texts.append(title + ' ' + plot)
        labels.append(genres)

# Tokenize and pad sequences
max_words = 10000  # Maximum number of words to consider in the tokenizer
max_length = 100   # Maximum length of the sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=max_length)

# Print the preprocessed data
print("Preprocessed Data:")
print("Texts:", texts[:5])
print("Labels:", labels[:5])
print("Tokenized Sequences:", sequences[:5])
print("Padded Data:", data[:5])
