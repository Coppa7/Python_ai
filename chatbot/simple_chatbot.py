from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from amazon_scraper import get_price_discount, name_url, synonims
import matplotlib.pyplot as plt
import json
import random
import pickle
import os
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def create_model(transformer, encoder):
    # Opening local folder to load data
    with open(os.path.join(os.path.dirname(__file__), "chatbot_qa.json"), "r", encoding='utf-8') as f:
        data = json.load(f)
    intents = data["intents"]   
    X = []
    y = []
    
    for intent in intents:
        for prompt in intent['prompts']:
            X.append(prompt)
            y.append(intent['tag'])
            
    # Splitting data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)

    # Transformer + encoding
    X_trainv = transformer.encode(X_train)
    X_valv = transformer.encode(X_val)


    y_trainl = encoder.fit_transform(y_train)
    y_vall = encoder.transform(y_val)

    # To be optimized
    early_stopping = EarlyStopping(
        min_delta = 0.001,
        patience=25,
        restore_best_weights=True,
    )

    # Neural network, to be optimized
    model = Sequential([
        #layers.Input(shape=(X_trainv.shape[1],)),  # input layer
        layers.Dense(128, activation='relu', input_shape=(X_trainv.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(encoder.classes_), activation='softmax')
    ])


    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    # Train model
    history = model.fit(
        X_trainv, y_trainl,
        validation_data=[X_valv, y_vall],
        batch_size = 32,
        epochs=500,
        callbacks=[early_stopping],
    )
    return history, model, intents



# Function to plot the model (training/validation loss and accuracy)
def plot_model(history):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Test function to check if the model works
def frase_input(intents, model, transformer, encoder):
    frase = ""
    while frase != "exit":
        frase = input()
        frasev = transformer.encode([frase])
        pred_probs = model.predict(frasev) # probability for each class 
        pred_intent = encoder.inverse_transform([pred_probs.argmax()])[0]
        print(pred_probs*100)
        print(pred_intent)
        if pred_probs[0].max() <= 0.50 or pred_intent == 'fallback':   
            print("I'm sorry, I couldn't understand.") 
            #Fallback phrase
            continue
        for intent in intents:
            if pred_intent == intent['tag'] and pred_intent != "price_request":
                # Needs to be changed to be more general
                print(random.choice(intent['responses']))
            elif pred_intent == "price_request":
                print(get_price_discount(name_url, synonims, frase))



transformer = SentenceTransformer('all-MiniLM-L6-v2')
encoder = LabelEncoder()
history, model, intents = create_model(transformer, encoder)

if __name__ == '__main__':
    plot_model(history)
    
model.export("chatbot_model")

frase_input(intents, model, transformer, encoder)


with open("transformer.pkl", "wb") as f:
    pickle.dump(transformer, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
    
    