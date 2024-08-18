import numpy as np
import gensim.downloader as api
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import pickle
import os

FILENAME = '../../data/rotten_tomatoes_movie_reviews.csv'
rows = []
fields = []
MODEL_PATH = '../../models/glove-wiki-gigaword-100.pkl'
REVIEW_TEXT_PATH = '../../data/review_text.pkl'
REVIEW_SENTIMENT_PATH = '../../data/review_sentiment.pkl'

if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'wb+') as file:
        model = api.load('glove-wiki-gigaword-100')
        print(f"[SAVING MODEL] Saving text embedding model at {MODEL_PATH}")
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING MODEL] Text embedding model successfully saved locally")
else:
    print(f"[MODEL FOUND] Found model at {MODEL_PATH}")
    with open(MODEL_PATH, 'rb+') as file:
        print(f"[LOADING MODEL] Attempting to load text embedding model from {MODEL_PATH}")
        model = pickle.load(file)
        print(f"[LOADING MODEL] Load successful")



def document_to_vector(doc, model):
    vectors = [model[word] for word in doc if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# Example data


if not os.path.exists(REVIEW_TEXT_PATH) or not os.path.exists(REVIEW_SENTIMENT_PATH):
    with open(FILENAME, 'r', encoding="utf8") as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        # num = 1
        for row in csvreader:
            # print(f"appending row {num}")
            rows.append(row)
            # num += 1


        # get total number of rows
        print("Total no. of rows: %d " % csvreader.line_num)

        # printing the field names
        print('Field names are:' + ', '.join(field for field in fields))

    for columns in rows[0]:
        print(columns)

    print(rows[1][8])
    print(rows[1][9])

    data = pd.read_csv(FILENAME)
    texts = []
    for i in range(1, 10000):
        texts.append(rows[i][8])

    labels = []

    for i in range(1, 10000):
        if rows[i][9] == 'POSITIVE':
            labels.append(1)
        else:
            labels.append(0)

    # Example labels

    # Preprocess and convert texts to vectors
    x = np.array([document_to_vector(text.lower().split(), model) for text in texts])
    y = np.array(labels)

    with open(REVIEW_TEXT_PATH, 'wb+') as file:
        print(f"[SAVING DATA] Saving review text at {REVIEW_TEXT_PATH}")
        pickle.dump(x, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING DATA] Review text successfully saved locally")

    with open(REVIEW_SENTIMENT_PATH, 'wb+') as file:
        print(f"[SAVING DATA] Saving review sentiments at {REVIEW_SENTIMENT_PATH}")
        pickle.dump(y, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING DATA] Review sentiments successfully saved locally")


else:
    print(f"[DATA FOUND] Found data at {REVIEW_TEXT_PATH}")
    print(f"[DATA FOUND] Found data at {REVIEW_SENTIMENT_PATH}")
    with open(REVIEW_TEXT_PATH, 'rb+') as file:
        print(f"[LOADING DATA] Attempting to load review text from {REVIEW_TEXT_PATH}")
        x = pickle.load(file)
        print(f"[LOADING DATA] Load successful")

    with open(REVIEW_SENTIMENT_PATH, 'rb+') as file:
        print(f"[LOADING DATA] Attempting to load review sentiments from {REVIEW_SENTIMENT_PATH}")
        y = pickle.load(file)
        print(f"[LOADING DATA] Load successful")


x = pd.DataFrame(x)
y = pd.Series(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


