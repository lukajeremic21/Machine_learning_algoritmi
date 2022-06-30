import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 

df = pd.read_csv('zadatak_4.csv')
# print(df.head())

# X DATASET BEZ ZADNJE KOLONE KOJA TREBA DA SE PREDVIDI
X = df.drop(['Da li je prevezen'], axis=1)
# Y KOLONA SOCIVA
y=df['Da li je prevezen']

# # DIJELJENJE VRIJEDNOSTI U TEST I TRENING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.75, random_state = 1)

# STRING VRIJEDNOSTI KONVERTOVATI U 0,1,2...
encoder = ce.OrdinalEncoder(cols=['Matična', 'Kabina', 'Destinacija', 'Starost', 'VIP', 'Ime'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_test.head())
y_train=y_train.astype('int')
# OBJEKAT ZA TRENING(criterion:entropy ili gini, default gini)
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3, random_state=0, splitter='best')

y_pred_en = clf_en.predict(X_test)

plt.figure(figsize=(12,8))
tree.plot_tree(clf_en.fit(X_train, y_train), feature_names=X_train.columns)