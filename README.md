# Movie Genre Classification using TensorFlow

This script demonstrates how to train a movie genre classification model using TensorFlow. The model classifies movies into three genres: Western, History, and Musical.

## Installation

1. Install Python (version 3.6 or above) if you haven't already.

2. Install TensorFlow using pip:

```
pip install tensorflow
```


3. Install the required additional modules:
```
pip install pymongo
```
## Usage

1. Make sure you have a MongoDB Atlas account set up. If not, sign up for free at [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas).

2. Clone this repository to your local machine or download the files.

3. Open the `movies.py` script in a text editor.

4. Replace `<your_connection_string>` with your actual MongoDB Atlas connection string in the `client = pymongo.MongoClient('<your_connection_string>')` line.

5. Open a terminal or command prompt and navigate to the project directory.

6. Run the script using the following command:
```
python3 movies.py
```

This will start the training process and display the progress and accuracy.

7. Once the model is trained, it will be saved as `movie_genre_model.keras` in the current directory.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
