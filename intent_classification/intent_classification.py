from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np




intents = pd.read_csv("C:/Users/franc/Desktop/ai_test/intent_classification/intenti.csv")
X = intents['Frase']
y = intents["Intento"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)
vectorizer = CountVectorizer()
encoder = LabelEncoder()

X_trainv = (vectorizer.fit_transform(X_train)).toarray()
X_valv = (vectorizer.transform(X_val)).toarray()

y_trainl = encoder.fit_transform(y_train)
y_vall = encoder.transform(y_val)

early_stopping = EarlyStopping(
    min_delta = 0.001,
    patience=5,
    restore_best_weights=True,
    
)

model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[X_trainv.shape[1]]),
    layers.Dense(16, activation='relu'),
    layers.Dense(4, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
    
)

history = model.fit(
    X_trainv, y_trainl,
    validation_data=[X_valv, y_vall],
    batch_size = 5,
    epochs=500,
    callbacks=[early_stopping],
)

'''
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss durante l\'allenamento')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy durante l\'allenamento')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
'''

frase = input()
frasev = vectorizer.transform([frase])
pred_probs = model.predict(frasev.toarray()) #probabilità per ciascuna classe
pred_intent = encoder.inverse_transform([pred_probs.argmax()])[0]
print(pred_probs*100)
print(pred_intent)





