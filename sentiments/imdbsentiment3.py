import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

reviews = pd.read_csv("C:/Users/franc/Desktop/ai_test/sentiments/IMDB_Dataset.csv")
X = reviews.review
y = reviews.sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipeline1 = Pipeline(steps=[('vectorizer', CountVectorizer()), ('model', RandomForestClassifier(n_estimators=50, random_state=0))])
pipeline1.fit(X_train, y_train)
encoder = LabelEncoder()

score = -1 * cross_val_score(pipeline1, X, encoder.fit_transform(y), cv=2, scoring='neg_mean_absolute_error')

print("Errore: ", score)

rev = "This movie was a complete waste of time. The plot made no sense, the acting was terrible, and I couldn’t wait for it to end."
pred = pipeline1.predict([rev])
print("Rev: ", rev, "\nPred: ", pred)
