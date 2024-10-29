
# Magic the Gathering AI
An AI program in Python that predicts the effect size of an MTG card by evaluating the text on the card, comparing it to the cost of actually playing the card to predict if the card is over- or undervalued for its potential effects.

## MTG cards
A magic card includes lots of data, some of which the AI can use; the rest is not included in the training data (i.e., the AI cannot "see" that information).

The mana value of the card (the "cost" of playing the card) is circled in red below. While a *player* can see this, the AI cannot. 

The green box below is the text that describes the effect of playing the card; this is the ONLY information the AI has access to. (It does not "read" the "flavor text" in italics on any card.)

The card data comes from a public API called [ScryFall](https://scryfall.com).

![049](https://github.com/BlaiseBaptist/MTG-AI/assets/40903991/78ea7ed4-84e2-419c-9273-3ade33d8bf79)

## XYZ


The model guesses what it thinks the mana value of the card should be based on all other cards this can indicate the power level of the card.

## How it works
The code works in a couple of discrete steps. it uses
 1. pandas to filter the initial set of all magic card to just the ones I want to train on
 2. a TfidfVectorizer to turn the words on the magic card in numbers
 3. a Ridge Regression to guess the mana value from the numbers   
The code uses pandas to filter the data from list of all magic cards

