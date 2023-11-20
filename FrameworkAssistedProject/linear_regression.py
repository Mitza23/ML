import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('bodyfat.csv')
# df.fillna(method='ffill', inplace=True)
x = np.array(df.loc[:, df.columns != 'BodyFat']).reshape(-1, 14)
y = np.array(df['BodyFat']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.coef_)
