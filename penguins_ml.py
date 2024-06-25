import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
st.title('Penguin Classifier: A Machine Learning App')
st.write("This app uses 6 inputs to predict the species of penguin using "
        "a model built on the Palmer Penguins dataset. Use the formbelow"
        " to get started!")
penguin_df = pd.read_csv('penguins.csv')
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()