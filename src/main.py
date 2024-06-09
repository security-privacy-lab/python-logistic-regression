from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression as CustomLogisticRegression
from data import x_train, x_test, y_train, y_test
import os
import pickle

MODEL_DIRECTORY = '../models'
MODEL_PATH = '../models/Custom_Logistic_Regression.pkl'

print(os.path.exists(MODEL_DIRECTORY))

if not os.path.exists(MODEL_PATH):
    print(f"[MODEL NOT FOUND] No model at {MODEL_PATH}")
    lr = CustomLogisticRegression()
    lr.fit(x_train, y_train, epochs=150)

    with open(MODEL_PATH, 'wb+') as file:
        print(f"[SAVING MODEL] Saving model at {MODEL_PATH}")
        pickle.dump(lr, file, pickle.HIGHEST_PROTOCOL)
        print(f"[SAVING MODEL] Model successfully saved locally")
else:
    print(f"[MODEL FOUND] Found model at {MODEL_PATH}")
    with open(MODEL_PATH, 'rb+') as file:
        print(f"[LOADING MODEL] Attempting to load model from {MODEL_PATH}")
        lr = pickle.load(file)
        print(f"[LOADING MODEL] Load successful")


pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print("Custom made Regression Accuracy:", accuracy)

model = LogisticRegression(solver='newton-cg', max_iter=150)
model.fit(x_train, y_train)
pred2 = model.predict(x_test)
accuracy2 = accuracy_score(y_test, pred2)
print("Built-in Regression Accuracy", accuracy2)
