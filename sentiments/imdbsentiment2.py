import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

reviews = pd.read_csv("C:/Users/franc/Desktop/ai_test/sentiments/IMDB_Dataset.csv")
X = reviews.review
y = reviews.sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = RandomForestClassifier()
vectorizer = CountVectorizer()
encoder = LabelEncoder()
X_trainv = vectorizer.fit_transform(X_train)
X_testv = vectorizer.transform(X_test)

model.fit(X_trainv, y_train)

pred = model.predict(X_testv)
y_test_encoded = encoder.fit_transform(y_test)
pred_encoded = encoder.transform(pred) 
abserr = mean_absolute_error(y_test_encoded, pred_encoded)
print("Review: ", X_test.iloc[0], "\nPred: ", pred[0], "\nSent: ", y_test.iloc[0], "Errore: ", abserr)