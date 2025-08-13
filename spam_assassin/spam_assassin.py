import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model = RandomForestClassifier(n_estimators=100, random_state=42)

email_data = pd.read_csv("spam_assassin.csv")
emails = email_data.text
y_train = email_data.target

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(emails)
model.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

#Testare il modello

email = """From: john.doe@example.com
Subject: Meeting Reminder

Hi team,

Just a quick reminder about our meeting tomorrow at 10 AM in the conference room.  
Please bring your reports and be prepared to discuss the project status.

Thanks,  
John
"""


email_vect = vectorizer.transform([email])
pred = model.predict(email_vect)
print(pred)


