import requests
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz

def get_price_discount(name_url, synonims, phrase):
    '''
    Uses scraping to find the price and discount (if there is one) of an item on amazon.it.
    This scraping is done only for learning purposes. A request is only sent once the program
    is ran. There's no multiple requests.
    '''
    
    phrase = str(phrase).lower()
    word, id_ret = similar_name(phrase, name_url, synonims) # id ret is the id of the item found by the function
    if id_ret == -1:
        print("Product not found")
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
    try:
        # Test for exception
        # url = "https://example.com/invalid" 
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()        
    except requests.RequestException as e:
        print(f"Error: {e}")
        return
    
    # Possible to add a check for captchas 
    
    
    html = response.text 
    soup = BeautifulSoup(html, "html.parser")

    # Finding the price (unstable if the HTML page changes, but it seems to be pretty stable so I wont change it
    # for now
    price = soup.find("span", class_="a-offscreen")
    discount = soup.find("span", class_="a-size-large a-color-price savingPriceOverride aok-align-center reinventPriceSavingsPercentageMargin savingsPercentage")

    if price:
        price = price.get_text(strip=True)
    else:
        price = "Price not found"
        
    if discount:
        discount = discount.get_text(strip=True)
    else:
        discount = "Discount not found"
        
    print(f"Price: {price}")
    print(f"Discount:  {discount}")
    
    


def similar_name(phrase, name_url, synonims):
    
    '''
    This function sees if the phrase written by the user contains the name of one of the
    objects included in the dictionary. If not, it searches for the most similar object name
    in the synonims dictionary.
    '''
    
    best_match1 = process.extractOne(phrase, list(name_url.keys()), scorer=fuzz.token_set_ratio)
    best_match2 = process.extractOne(phrase, list(synonims.keys()), scorer=fuzz.token_set_ratio)
    
    # I arbitrarily set 85 as the "worst accuracy score possible to still count as a phrase" for the object
    # Can be changed, needs testing
    

    if best_match1[1] >= 85:    
        print(f"Certainty: {best_match1[1]}")    
        return best_match1[0], 1      
    if best_match2[1] >= 85:
        print(f"Certainty: {best_match2[1]}")
        return best_match2[0], 2
    return None, -1
    

# phrase to url dictionary (on italian amazon)
name_url = {
    "raspberry pi 5": "https://www.amazon.it/dp/B0CK3L9WD3",
    "tastiera logitech": "https://www.amazon.it/dp/B07W6GVS5C"
}

# "synonims" dictionary (on italian amazon)

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
