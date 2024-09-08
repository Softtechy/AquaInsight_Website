import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('testcopy.csv')
print(df.head())


sns.lmplot(x='Latitude', y='level', data=df)  
sns.lmplot(x='Longitude', y='level', data=df)  


x_df = df.drop('level', axis=1)
y_df = df['level']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

from sklearn import linear_model


model = linear_model.LinearRegression()



model.fit(X_train, y_train)
print(model.score(X_train, y_train))  


prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)

import pickle
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[20.1, 56.3]]))
