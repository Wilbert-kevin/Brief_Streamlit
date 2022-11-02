#Import des differents modules necessaires
import numpy as  np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from streamlit.elements.button import ButtonSerde
import streamlit as st

# Lecture du fichier
df_Iris = pd.read_csv('iris.csv')

# entrainement par les donnees
knn = KNeighborsClassifier(n_neighbors=5)
X = df_Iris.drop("species", axis=1)
y = df_Iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
sns.pairplot(df_Iris, hue = 'species')
sns.set_theme(style="ticks")
knn.fit(X_train, y_train)

# Titre de mon fichier
st.title('Quel iris suis-je')
st.subheader('Choisissez vos parametres !')

# Sliders min - max
longueur_sepale = st.sidebar.slider("Choisissez la longueur de la sépale", 4.0, 8.0)
largeur_sepale = st.sidebar.slider("Choisissez la largeur de la sépale", 2.0, 5.0)
longueur_petale = st.sidebar.slider("Choisissez la longueur de la pétale", 1.0, 8.0)
largeur_petale = st.sidebar.slider("Choisissez la largeur de la pétale", 0.0, 3.0)

# Especes de l'utilisateur
data_utilisateur = [longueur_sepale, largeur_sepale, longueur_petale, largeur_petale]
espece_utilisateur = knn.predict([data_utilisateur])

# choix utilisateur avec utilisation de dicotionnaire et graphique
point_en_cours = {
        'sepal_length': [longueur_sepale],
        "sepal_width" : [largeur_sepale],
        "petal_length" : [longueur_petale],
        "petal_width": [largeur_petale],
        "species": "Votre point"
        }


df_choix_utilisateur = pd.DataFrame(point_en_cours)
df_final = pd.concat([df_Iris, df_choix_utilisateur], axis=0)

# Affichage du bouton Validé
if st.button("Validé"):
    st.success("Félicitation " + espece_utilisateur[0])
    st.pyplot(sns.pairplot(df_final, x_vars=["petal_length"], y_vars=["petal_width"], hue="species", markers=["o", "s", "D", "p"]))
    st.pyplot(sns.pairplot(df_final, x_vars=["sepal_length"], y_vars=["sepal_width"], hue="species", markers=["o", "s", "D", "p"]))
    st.write('Bravo')
    st.write('')