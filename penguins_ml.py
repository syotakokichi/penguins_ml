import pickle  # pickleをインポートしています。モデルを保存・読み込みに使用します。
import matplotlib.pyplot as plt  # プロットのためのmatplotlibをインポートします。
import pandas as pd  # データ処理のためのpandasをインポートします。
import seaborn as sns  # グラフ作成のためのseabornをインポートします。
import streamlit as st  # Streamlitアプリケーションの構築に必要なstreamlitをインポートします。
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレストモデルをインポートします。
from sklearn.metrics import accuracy_score  # モデルの評価に使用するaccuracy_scoreをインポートします。
from sklearn.model_selection import train_test_split  # データをトレーニングとテストに分割するためのtrain_test_splitをインポートします。

st.title("Penguin Classifier")  # アプリケーションのタイトルを設定しています。

st.write(
    """This app uses 6 inputs to predict
    the species of penguin using a model
    built on the Palmer's Penguins dataset.
    Use the form below to get started!"""
)  # アプリケーションの説明を記述しています。

password_guess = st.text_input("What is the Password?")  # パスワード入力を取得しています。
if password_guess != "streamlit_is_great":  # 入力されたパスワードが正しくない場合は停止します。
    st.stop()

penguin_file = st.file_uploader("Upload your own penguin data")  # ユーザーがアップロードしたペンギンデータを受け取ります。

if penguin_file is None:  # ファイルがアップロードされていない場合
    # ファイルがない場合はエラーになります。
    rf_pickle = open("random_forest_penguin.pickle", "rb")
    map_pickle = open("output_penguin.pickle", "rb")
    rfc = pickle.load(rf_pickle)  # ピクルスからランダムフォレストモデルをロードします。
    unique_penguin_mapping = pickle.load(map_pickle)  # ピクルスからユニークなペンギンマッピングをロードします。
    rf_pickle.close()
    map_pickle.close()
    penguin_df = pd.read_csv("penguins.csv")  # ローカルのCSVファイルからデータを読み込みます。
else:
    penguin_df = pd.read_csv(penguin_file)  # ユーザーがアップロードしたCSVファイルからデータを読み込みます。
    penguin_df = penguin_df.dropna()  # 欠損値を削除します。
    output = penguin_df["species"]
    features = penguin_df[
        [
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
    ]
    features = pd.get_dummies(features)  # カテゴリカル変数をダミー変数に変換します。
    output, unique_penguin_mapping = pd.factorize(output)  # ターゲット変数を数値に変換します。
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8)  # データをトレーニングセットとテストセットに分割します。
    rfc = RandomForestClassifier(random_state=15)  # ランダムフォレストモデルのインスタンスを作成します。
    rfc.fit(x_train, y_train)  # モデルをトレーニングします。
    y_pred = rfc.predict(x_test)  # テストデータで予測を行います。
    score = round(accuracy_score(y_pred, y_test), 2)  # モデルの精度を計算します。
    st.write(
        f"""We trained a Random Forest model on these data,
        it has a score of {score}! Use the
        inputs below to try out the model"""
    )  # モデルの学習結果を表示します。

with st.form("user_inputs"):  # ユーザーの入力フォームを作成します。
    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])  # 島の選択ボックスを表示します。
    sex = st.selectbox("Sex", options=["Female", "Male"])  # 性別の選択ボックスを表示します。
    bill_length = st.number_input("Bill Length (mm)", min_value=0)  # 嘴の長さの数値入力を表示します。
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)  # 嘴の深さの数値入力を表示します。
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)  # フリッパーの長さの数値入力を表示します。
    body_mass = st.number_input("Body Mass (g)", min_value=0)  # 体の重さの数値入力を表示します。
    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1
# 島の選択に基づいて、ダミー変数を設定しています。

sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1
# 性別の選択に基づいて、ダミー変数を設定しています。

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
)  # 新しい入力に対して、ペンギンの種を予測します。
st.subheader("Predicting Your Penguin's Species:")
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f"We predict your penguin is of the {prediction_species} species")  # 予測されたペンギンの種を表示します。

st.write(
    """We used a machine learning
    (Random Forest) model to predict the
    species, the features used in this
    prediction are ranked by relative
    importance below."""
)  # モデルの特徴の重要度を表示します。

st.image("feature_importance.png")  # 特徴の重要度を示す画像を表示します。

st.write(
    """Below are the histograms for each
continuous variable separated by penguin species.
The vertical line represents the inputted value."""
)  # 各連続変数のヒストグラムを表示します。

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_length_mm"], hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Bill Length by Species")
st.pyplot(ax)  # 嘴の長さの種ごとのヒストグラムを表示します。

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_depth_mm"], hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Bill Depth by Species")
st.pyplot(ax)  # 嘴の深さの種ごとのヒストグラムを表示します。

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["flipper_length_mm"], hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Flipper Length by Species")
st.pyplot(ax)  # フリッパーの長さの種ごとのヒストグラムを表示します。