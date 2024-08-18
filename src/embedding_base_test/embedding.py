from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from data.rotten_movie_word_sentiment import *
from logistic_regression.model import LogisticRegression as custom_regression
import pickle



lr = LogisticRegression()
lr.fit(x_train, y_train)
print("trained built-in model")

custom_lr = custom_regression()
custom_lr.fitVerbose(x_train, y_train, epochs=150)
print("trained custom model")

pred1 = custom_lr.predict(x_test)
pred2 = lr.predict(x_test)
print("finished predictions")

accuracy1 = accuracy_score(y_test, pred1)
accuracy2 = accuracy_score(y_test, pred2)

print(f"Accuracy of custom model:   {accuracy1}")
print(f"Accuracy of built-in model: {accuracy2}")
