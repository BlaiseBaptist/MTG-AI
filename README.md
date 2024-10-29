
# Magic the Gathering AI
An AI program in Python that predicts the effect size of an MTG card by evaluating the text on the card, comparing it to the cost of actually playing the card to predict if the card is over- or undervalued for its potential effects.

## MTG cards
A Magic card includes lots of data, some of which the AI can use; the rest is not included in the training data (i.e., the AI cannot "see" that information).

The mana value of the card (the "cost" of playing the card) is circled in red below. While a *player* can see this, the AI cannot. 

The green box below is the text that describes the effect of playing the card; this is the ONLY information the AI has access to. (It does not "read" the "flavor text" in italics on any card.)

The card data comes from a public API called [ScryFall](https://scryfall.com).

![049](https://github.com/BlaiseBaptist/MTG-AI/assets/40903991/78ea7ed4-84e2-419c-9273-3ade33d8bf79)

## How it works
The code works in a couple of discrete steps:
 - It uses pandas to simplify the data set by filtering out creature and land cards; this removes about 50% of the cards.
 - Next, it uses pandas to filter out defunct / banned / "un"-cards to clean up the data set (while there are not very many cards in this category, they would have large unwanted effect on the model).
 - It uses a TfidfVectorizer to parse the language (i.e., to turn the words on the Magic card into numbers) so the AI can read it.
 - Then it runs a ridge regression, making a model able to predict the mana value of a particular card from these numbers. 
 - Now the program will display a random card and tell you its prediction for the mana cost, given the words on the card, and will also show you its actual mana cost. The user can then compare these two numbers to judge if the card is worth its mana cost or is "underpowered."
 
 ## Further development possibilities 
 
 - Make it able to work with creature cards: able to see and account for power (damage) and toughness (health) features on the card. The ridge regression is not built to handle more than one feature, so I would need to either use multiple regressions or find a new regression type to add this complexity.  
 - Be able to export the model so it doesn't have to be re-trained every time I'd like to use it.
 - Include newly-released card sets in the data, because Magic releases new sets of cards all the time. 

