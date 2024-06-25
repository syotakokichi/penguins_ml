# pandasライブラリをインポート
import pandas as pd

# scikit-learnのaccuracy_score関数をインポート
from sklearn.metrics import accuracy_score

# scikit-learnのRandomForestClassifierクラスをインポート
from sklearn.ensemble import RandomForestClassifier

# scikit-learnのtrain_test_split関数をインポート
from sklearn.model_selection import train_test_split

# pickleライブラリをインポート（モデルの保存に使用）
import pickle

# 'penguins.csv'ファイルを読み込み、データフレームとして保存
penguin_df = pd.read_csv('penguins.csv')

# 欠損値を含む行を削除
penguin_df.dropna(inplace=True)

# 'species'列を目的変数として抽出
output = penguin_df['species']

# 特徴量として指定した列を抽出
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]

# カテゴリ変数をダミー変数に変換
features = pd.get_dummies(features)

# 目的変数を数値に変換
output, uniques = pd.factorize(output)

# データを訓練用とテスト用に分割（訓練用20%、テスト用80%）
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)

# ランダムフォレスト分類器を作成
rfc = RandomForestClassifier(random_state=15)

# 訓練データを使ってモデルを訓練
rfc.fit(x_train.values, y_train)

# テストデータを使って予測
y_pred = rfc.predict(x_test.values)

# モデルの正解率を計算
score = accuracy_score(y_pred, y_test)

# 正解率を出力
print('Our accuracy score for this model is {}'.format(score))

# ランダムフォレストモデルをpickleファイルに保存
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

# 目的変数のユニーク値をpickleファイルに保存
output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()