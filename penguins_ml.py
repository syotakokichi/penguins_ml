import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# タイトルを設定
st.title('Penguin Classifier: A Machine Learning App')

# 説明文を表示
st.write("This app uses 6 inputs to predict the species of penguin using "
        "a model built on the Palmer Penguins dataset. Use the form below "
        "to get started!")

password_guess = st.text_input('What is the Password?')
if password_guess != 'streamlit_password':
  st.stop()

# データを読み込む
penguin_file = st.file_uploader('Upload your own penguin data')

# モデルとマッピングを読み込む
with open('random_forest_penguin.pickle', 'rb') as rf_pickle:
    rfc = pickle.load(rf_pickle)

with open('output_penguin.pickle', 'rb') as map_pickle:
    unique_penguin_mapping = pickle.load(map_pickle)

# フォームを作成し、ユーザー入力を受け取る
with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.form_submit_button()

# 島の選択に基づいてフラグを設定
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1

# 性別の選択に基づいてフラグを設定
sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

# ユーザーの入力に基づいて予測を行う
new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male,
        ]
    ]
)

# 予測されたペンギンの種類を取得
prediction_species = unique_penguin_mapping[new_prediction][0]

# 予測結果を表示
st.write(f"We predict your penguin is of the {prediction_species} species")