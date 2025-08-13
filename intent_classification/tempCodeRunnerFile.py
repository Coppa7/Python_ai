import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers



intents = pd.read_csv("C:/Users/franc/Desktop/ai_test/intent_classification/intenti.csv")
print(intents.head())
X = intents['Frase']
y = intents["Intento"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=0)
vectorizer = CountVectorizer()
encoder = LabelEncoder()

X_trainv = vectorizer.fit_transform(X_train)
X_valv = vectorizer.transform(X_val)

y_trainl = encoder.fit_transform(y_train)
y_vall = encoder.transform(y_val)

print("Dio porco")



