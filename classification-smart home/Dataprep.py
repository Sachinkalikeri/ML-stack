import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import itertools
import math

data = pd.read_csv("/kaggle/input/smart-home-commands-dataset/dataset.csv")
del data["Number"]

sentences = data['Sentence']
categories = data['Category']
subcategories = data['Subcategory']
actions = data['Action']

uniquecategories = list(set(categories))
uniquesubcategories = list(set(subcategories))
uniqueactions = list(set(actions))

vocabulary = list(set(itertools.chain.from_iterable([word_tokenize(sentence.lower()) for sentence in sentences])))

categoryvectors = [uniquecategories.index(w) for w in categories]
subcategoryvectors = [uniquesubcategories.index(w) for w in subcategories]
actionvectors = [uniqueactions.index(w) for w in actions]

sentencevectors = []
for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence.lower())
    vector = [1 if word in tokenized_sentence else 0 for word in vocabulary]
    sentencevectors.append(vector)

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(sentencevectors, categoryvectors, test_size=0.25, random_state=42)
X_train_subcat, X_test_subcat, y_train_subcat, y_test_subcat = train_test_split(sentencevectors, subcategoryvectors, test_size=0.25, random_state=42)
X_train_action, X_test_action, y_train_action, y_test_action = train_test_split(sentencevectors, actionvectors, test_size=0.25, random_state=42)

