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
    legal_training_cards = ((cards.loc[cards['legalities'] != {}])[["text", "convertedManaCost"]]).replace(np.nan, "")
    #print("card text and cost: ",legal_training_cards.iloc[[4477]])
    Tfidf = TfidfVectorizer()
    X_train, X_test = train_test_split(Tfidf.fit_transform(legal_training_cards["text"]))
    Y_train, Y_test = train_test_split(legal_training_cards["convertedManaCost"])
    clf = RidgeClassifier()
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    print("card pred cost")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    main()

