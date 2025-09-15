from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from amazon_scraper import get_price_discount, name_url, synonims
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import pickle


def create_model(vectorizer, encoder):
    with open("C:/Users/franc/Desktop/ai_test/chatbot/chatbot_qa.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    intents = data["intents"]   
    X = []
    y = []
    for intent in intents:
        for prompt in intent['prompts']:
            X.append(prompt)
            y.append(intent['tag'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)

    X_trainv = (vectorizer.fit_transform(X_train)).toarray()
    X_valv = (vectorizer.transform(X_val)).toarray()

    y_trainl = encoder.fit_transform(y_train)
    y_vall = encoder.transform(y_val)

    early_stopping = EarlyStopping(
        min_delta = 0.001,
        patience=30,
        restore_best_weights=True,
    )

    model = keras.Sequential([
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', input_shape=[X_trainv.shape[1]]),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
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
        batch_size = 32,
        epochs=500,
        callbacks=[early_stopping],
    )
    return history, model, intents




def plot_model(history):
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

def frase_input(vectorizer, encoder):
    frase = input()
    frasev = vectorizer.transform([frase])
    pred_probs = model.predict(frasev.toarray()) # probability for each class 
    pred_intent = encoder.inverse_transform([pred_probs.argmax()])[0]
    print(pred_probs*100)
    print(pred_intent)
    for intent in intents:
        if pred_intent == intent['tag'] and pred_intent != "richiesta_prezzo": # Needs to be changed to be more general
            print(random.choice(intent['responses']))
        elif pred_intent == "richiesta_prezzo":
            get_price_discount(name_url, synonims, frase)
            break



vectorizer = CountVectorizer()
encoder = LabelEncoder()
history, model, intents = create_model(vectorizer, encoder)

plot_model(history)

model.save("chatbot_model.h5")

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
    
    