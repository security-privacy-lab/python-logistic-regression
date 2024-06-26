import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

import numpy as np


def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')


    return x, y







x, y = sklearn_to_df(load_breast_cancer())


ss = StandardScaler()
x_scaled = pd.DataFrame(ss.fit_transform(x),columns = x.columns)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

x_scaled_train, x_scaled_test = train_test_split(x_scaled, test_size=0.2, random_state=42)

q = 1
