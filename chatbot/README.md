## Simple Chatbot using Neural Network

- Different programs to make a simple chatbot, which can chat with the user by understanding what he says and replying back with different answers.

## Functions

The primary function of the chatbot is to chat with the user in a basic manner ( replying to greetings, compliments), but it also has other capabilities,
as it can give the user information about the price and discount value of certain objects on Amazon.it. 

## How it works

The chatbot is trained using a Neural Network (which, for the most part, I have optimized to the best I could, although it can probably still be better)
and a dataset (chatbot_qa.json, manually created) which contains:

-User prompts which relate to a specific intent
-Aswers to such prompts 

The NN is built using only two hidden layers, with 128 and 64 neurons, due to the small dataset, and uses many different technologies to maximize
accuracy and optimize the model, such as dropout layers, early stopping, as well as:

-A sentence transformer, to transform the prompts into vectors used to train and validate the model
-Label encoder, to address each type of intent 
-Adam optimizer, used to minimize the Loss function
-Sparse categorical crossentropy, which is the loss function that allows each intent predicted by the model to have a "certainty" value, meaning
how certain the model is of that choice. This allows for the usage of a fallback based on this information.

The model, as previously stated, is also capable of searching for the price of an object on Amazon.it (.it for the Italian site), although
this only works for the specific objects which are found in the amazon_scraper.py file. The reason for the fact that the chatbot is unable to search
on Amazon on its own is due to the legality issue of scraping and because I just wanted to have a chatbot to which i could ask: "What is the price of X
right now" and getting a specific Amazon page without having to worry too much if its the "right one". This, however, could be updated in the future,
allowing to search for price and discount on Amazon (in a more freely matter), but also on other sites, comparing their price and finding the cheapest
one, or the one with most discount value.

## Testing the model

The model is saved onto a file, and it is then supposed to be used either in the training program by itself (there's many test prompts to check
the accuracy of the model, which goes from 94%-96% most of the time), or in a Flask webapp, which still needs to be worked on more thoroughly. 
The training program also allows to plot the Accuracy and Loss values, both for training and validation data, using matplotlib.

## Features in the future

Some of the features I intend to add in the future are:

-Porting the whole project onto a raspberry pi 
-Hosting the Flask webapp onto an online accessible webpage
-Allowing the chatbot to search for prices on more sites other than Amazon and in a way that's more general
-...

## Legality of scraping issue

The chatbot uses a scraping script which sends a limited amount of requests to Amazon.it to find the price and discount of an object. The scraping is 
done solely for a didactic purpose (I found it interesting to at least find it out how it worked), and it isn't intended to be used in a way that
is harmful to Amazon.
