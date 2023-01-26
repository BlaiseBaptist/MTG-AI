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
    legal_training_cards = ((cards.loc[cards['legalities'] != {}])[["text", "convertedManaCost","types","power","toughness","name"]]).replace(np.nan, "")
    Tfidf = TfidfVectorizer()
    X_train, X_test = train_test_split(Tfidf.fit_transform(legal_training_cards["text"]))
    Y_train, Y_test = train_test_split(legal_training_cards["convertedManaCost"])
    clf = RidgeClassifier()
    clf.fit(X_train, Y_train)
    while 0==0:
        print("\n\n\n")
        card = cards.iloc[[random.randrange(legal_training_cards.shape[0])]]
        print("card name",card["name"].values[0])
        print("creature stats: ",card['power'].values[0],"/",card['toughness'].values[0],sep='')
        print("card cost:",card['convertedManaCost'].values[0])
        print(card['text'].values[0])
        pred = clf.predict(Tfidf.transform(card['text']))
        print("card pred cost",pred[0])
        input("enter to go to next card")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    main()

