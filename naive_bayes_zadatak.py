# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import category_encoders 
from sklearn.preprocessing import OrdinalEncoder

# Importing the dataset
dataset = pd.read_csv('naive_bayes_zadatak.csv')

enc = OrdinalEncoder()
enc.fit(dataset[['Pol','Predmet', 'Zadovoljstvo roditelja', 'Broj dana odsustva', 'Klasa uspješnosti']])
dataset[['Pol','Predmet', 'Zadovoljstvo roditelja', 'Broj dana odsustva', 'Klasa uspješnosti']] = enc.transform(dataset[['Pol','Predmet', 'Zadovoljstvo roditelja', 'Broj dana odsustva', 'Klasa uspješnosti']])


print(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# print(X)
# print(y)

# Dijeljenje modela u train i test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

# Standardizovanje podataka odnosno skaliranje 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Treniranje modela
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# # Predikcija rezultata
# Zbog encodinga, odnosno konvertovanja stringa u int, date vrijednosti su 0 i 1 gdje je string. Primjer: Zadovoljstvo roditelja: G/B == 1/0
# Predikcija na osnovu:
#  𝑋 = {Pol = Ženski, Predmet = IT, Preuzeti resursi = 35, Pročitana obavještenja = 30, Zadovoljstvo roditelja = Good, Dani odsustva  = Under-7}
print(classifier.predict(sc.transform([[0, 1, 35, 30, 1, 1, 1]])))

# Predikcija testnih rezultata
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Confusion matrica
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

