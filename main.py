# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import random

# Use a breakpoint in the code line below to debug your script.


# Press the green button in the gutter to run the script.


def main():
    cards = pd.read_json("AllCards.json", orient='index')
    cards.types = (cards.types.map(lambda x: x[0]))
    cards = cards.loc[cards.types != "Creature"]
    cards = cards.loc[cards.types != "Land"]
    cards = cards.loc[cards.types != "Vanguard"]
    cards = cards.loc[cards.types != "Plane"]
    legal_training_cards = ((cards.loc[cards['legalities'] != {}])[["text", "convertedManaCost","types","power","toughness","name"]]).replace(np.nan, "")
    Tfidf = TfidfVectorizer()
    x_train, x_test = train_test_split(Tfidf.fit_transform(legal_training_cards["text"]))
    y_train, y_test= train_test_split(legal_training_cards["convertedManaCost"])
    clf = RidgeClassifier()
    clf.fit(x_train, y_train)
    while True:
        print("\n\n\n")
        card = cards.iloc[[random.randrange(legal_training_cards.shape[0])]]
        print("card name",card["name"].values[0])
        print("card type",card["types"].values[0])
        print("creature stats: ",card['power'].values[0],"/",card['toughness'].values[0],sep='')
        print("card cost:",card['convertedManaCost'].values[0])
        print(card['text'].values[0])
        pred = clf.predict(Tfidf.transform(card['text']))
        print("card pred cost",pred[0])
        input("enter to go to next card")
if __name__ == '__main__':
    main()

