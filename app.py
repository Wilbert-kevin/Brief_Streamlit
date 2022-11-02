# On import les librairie
import streamlit as st
import numpy as  np
import sklearn as svm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#on recupere le csv
df_Iris = pd.read_csv('iris.csv')

#
X = df_Iris.drop("species", axis=1)
y = df_Iris["species"]
knn = KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn.fit(X_train, y_train)
sns.pairplot (df_Iris, hue= "species")
sns.set_theme()


# Ajout un titre 
st.title('Iris ')

# 
st.subheader ('Quelle parametre choisissez vous ?')

# Sliders 
sp_length = st.slider('sepal_length', min_value=4.00, max_value=10.00)
sp_width  = st.slider('sepal_width', min_value=2.00, max_value=5.00)
sp_length = st.slider('petal_length', min_value=1.00, max_value=8.00)
sp_width  = st.slider('petal_width', min_value=0.00, max_value=2.50)


#
point_en_cours = {
        'sepal_length':  [sp_length],
        "sepal_width" :  [sp_width],
        "petal_length" : [sp_length],
        "petal_width":   [sp_width ],
        "species": "unknown"
        }


#choix 
df_choix_utilisateur = pd.DataFrame(point_en_cours)
df_final = pd.concat([df_Iris, df_choix_utilisateur], axis=0)


# Ajout du bouton "Validé" 
if st.button('Validé'):
    st.success('Bien joue')
    st.write('Felicitation')



