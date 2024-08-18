import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pickle

model = SentenceTransformer('all-mpnet-base-v2')

FILENAME = '../../data/rotten_tomatoes_movie_reviews.csv'
DATA_DIRECTORY = '../../data/rotten_tomatoes/all-mpnet-base-v2'
REVIEW_TEXT_TRAIN_PATH = '../../data/rotten_tomatoes/all-mpnet-base-v2/review_text_train.pkl'
REVIEW_TEXT_TEST_PATH = '../../data/rotten_tomatoes/all-mpnet-base-v2/review_text_test.pkl'
REVIEW_SENTIMENT_TRAIN_PATH = '../../data/rotten_tomatoes/all-mpnet-base-v2/review_sentiment_train.pkl'
REVIEW_SENTIMENT_TEST_PATH = '../../data/rotten_tomatoes/all-mpnet-base-v2/review_sentiment_test.pkl'

if ((not os.path.exists(REVIEW_TEXT_TRAIN_PATH))
        or (not os.path.exists(REVIEW_TEXT_TEST_PATH))
        or (not os.path.exists(REVIEW_SENTIMENT_TRAIN_PATH))
        or (not os.path.exists(REVIEW_SENTIMENT_TEST_PATH))):

    print(f"[DATA NOT FOUND] No data found at some location within {DATA_DIRECTORY}")
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    randomized_dataset = dataset.shuffle()

    x_train = model.encode(randomized_dataset['train']['text'])
    y_train = randomized_dataset['train']['label']
    x_test = model.encode(randomized_dataset['test']['text'])
    y_test = randomized_dataset['test']['label']

    with open(REVIEW_TEXT_TRAIN_PATH, 'wb+') as file:
        print(f"[SAVING DATA] Saving data at {REVIEW_TEXT_TRAIN_PATH}")
        pickle.dump(x_train, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING DATA] Data successfully saved locally")

    with open(REVIEW_TEXT_TEST_PATH, 'wb+') as file:
        print(f"[SAVING DATA] Saving data at {REVIEW_TEXT_TEST_PATH}")
        pickle.dump(x_test, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING DATA] Data successfully saved locally")

    with open(REVIEW_SENTIMENT_TRAIN_PATH, 'wb+') as file:
        print(f"[SAVING DATA] Saving data at {REVIEW_SENTIMENT_TRAIN_PATH}")
        pickle.dump(y_train, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING DATA] Data successfully saved locally")

    with open(REVIEW_SENTIMENT_TEST_PATH, 'wb+') as file:
        print(f"[SAVING DATA] Saving data at {REVIEW_SENTIMENT_TEST_PATH}")
        pickle.dump(y_test, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING DATA] Data successfully saved locally")
else:
    print(f"[DATA FOUND] All data found within {DATA_DIRECTORY}")

    with open(REVIEW_TEXT_TRAIN_PATH, 'rb+') as file:
        print(f"[LOADING DATA] Attempting to load data from {REVIEW_TEXT_TRAIN_PATH}")
        x_train = pickle.load(file)
        print(f"[LOADING DATA] Load successful")

    with open(REVIEW_TEXT_TEST_PATH, 'rb+') as file:
        print(f"[LOADING DATA] Attempting to load data from {REVIEW_TEXT_TEST_PATH}")
        x_test = pickle.load(file)
        print(f"[LOADING DATA] Load successful")

    with open(REVIEW_SENTIMENT_TRAIN_PATH, 'rb+') as file:
        print(f"[LOADING DATA] Attempting to load data from {REVIEW_SENTIMENT_TRAIN_PATH}")
        y_train = pickle.load(file)
        print(f"[LOADING DATA] Load successful")

    with open(REVIEW_SENTIMENT_TEST_PATH, 'rb+') as file:
        print(f"[LOADING DATA] Attempting to load data from {REVIEW_SENTIMENT_TEST_PATH}")
        y_test = pickle.load(file)
        print(f"[LOADING DATA] Load successful")


print("swag")