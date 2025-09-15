from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from amazon_scraper import get_price_discount, name_url, synonims
import pickle
import json
import random

model = load_model("chatbot_model.h5")
with open("C:/Users/franc/Desktop/ai_test/chatbot/chatbot_qa.json", "r", encoding='utf-8') as f:
        data = json.load(f)
intents = data["intents"] 

# Carica vectorizer e encoder
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('website.html')

@app.route('/chat', methods=['POST'])
def chat():
    frase = request.form['input_txt']
    frasev = vectorizer.transform([frase])
    pred_probs = model.predict(frasev.toarray()) # probability for each class 
    pred_intent = encoder.inverse_transform([pred_probs.argmax()])[0]
    #print(pred_probs*100)
    #print(pred_intent)
    for intent in intents:
        if pred_intent == intent['tag'] and pred_intent != "richiesta_prezzo": # Needs to be changed to be more general
            return render_template('website.html', response=random.choice(intent['responses']))
        elif pred_intent == "richiesta_prezzo":
            return render_template('website.html', response=get_price_discount(name_url, synonims, frase))



if __name__ == '__main__':
    app.run(debug=True)