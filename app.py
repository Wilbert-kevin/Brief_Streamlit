import streamlit as st
import numpy as  np
import sklearn as svm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


st.title(' Quel iris ')

st.slider('sepal_length')
st.slider('sepal_width')

st.slider('petal_length')
st.slider('petal_width')



