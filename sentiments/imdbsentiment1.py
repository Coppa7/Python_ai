import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

reviews = pd.read_csv("C:/Users/franc/Desktop/ai_test/sentiments/IMDB_Dataset.csv")
X = reviews.review
y = reviews.sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier()
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
model.fit(X_train_vect, y_train)
X_test_vect = vectorizer.transform(X_test)
predictions = model.predict(X_test_vect)
print("Testo:", X_test.iloc[0])
print("Vera etichetta:", y_test.iloc[0])
print("Predizione:", predictions[0])
 

