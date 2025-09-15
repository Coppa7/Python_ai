import requests
import time
import re
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz

def get_price_discount(name_url, synonims, phrase):
    phrase = str(phrase).lower()
    word, id_ret = similar_name(phrase, name_url, synonims) 
    if id_ret == -1:
        print("Prodotto non trovato")
        return
    elif id_ret == 1:
        url = name_url[word]
    else:
        url = name_url[synonims[word]]
        

    # Header for the HTTP request
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0 Safari/537.36",
    "Accept-Language": "it-IT,it;q=0.9"
    }

    response = requests.get(url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # Finding the price 
    price = soup.find("span", class_="a-offscreen")
    discount = soup.find("span", class_="a-size-large a-color-price savingPriceOverride aok-align-center reinventPriceSavingsPercentageMargin savingsPercentage")

    if price:
        price = price.get_text(strip=True)
    else:
        price = "Prezzo non trovato"
        
    if discount:
        discount = discount.get_text(strip=True)
    else:
        discount = "Sconto non trovato"
        
    print(f"Prezzo: {price}")
    print(f"Sconto:  {discount}")
    
    
# This function sees if the phrase of the object written by the user is similar to any of the object phrases in the 
# name_url dictionary. Only gets executed if the phrase is not in the dictionary itself or in the synonims dictionary

def similar_name(phrase, name_url, synonims):
    best_match1 = process.extractOne(phrase, list(name_url.keys()), scorer=fuzz.partial_ratio)
    best_match2 = process.extractOne(phrase, list(synonims.keys()), scorer=fuzz.partial_ratio)
    
    # I arbitrarily set 85 as the "worst accuracy score possible to still count as a phrase" for the object
    # Can be changed, needs testing
    

    if best_match1[1] >= 85:    
        print(f"Sicurezza parola: {best_match1[1]}")    
        return best_match1[0], 1      
    if best_match2[1] >= 85:
        print(f"Sicurezza parola: {best_match2[1]}")
        return best_match2[0], 2
    return None, -1
    

# phrase to url dictionary 
name_url = {
    "raspberry pi 5": "https://www.amazon.it/dp/B0CK3L9WD3",
    "tastiera logitech": "https://www.amazon.it/dp/B07W6GVS5C"
}

# "synonims" dictionary

synonims = {
    "raspberry 5": "raspberry pi 5",
    "raspberry pi 5 4 gb": "raspberry pi 5",
    "raspberry": "raspberry pi 5", # Not general
    "tastiera gaming logitech": "tastiera logitech",
    "logitech g213": "tastiera logitech",
    "logitech prodigy": "tastiera logitech",
    "tastiera gaming g213": "tastiera logitech",
    "tastiera gaming logitech g213": "tastiera logitech",
    "logitech g213 prodigy tastiera gaming": "tastiera logitech",
    "tastiera": "tastiera logitech" # Not general
}