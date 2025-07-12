from sklearn.model_selection import learning_curve
import streamlit as st
import pandas as pd
import joblib

st.title("体脂肪率変化予測アプリ【プラスマイナス0.25%版】")

try:
    clf_model = joblib.load("clf_model.pkl")
    input_columns = joblib.load("input_columns.pkl")
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

st.write("以下の入力から、体脂肪率が【増加・維持・減少】するかを予測します。")

user_input = {}
# 適切な初期値と範囲を設定
for col in input_columns:
    if "体重" in col:
        user_input[col] = st.slider(col, min_value=40.0, max_value=100.0, value=65.0, step=0.1)
    elif "基礎代謝" in col:
        user_input[col] = st.slider(col, min_value=1200.0, max_value=2500.0, value=1500.0, step=5.0)
    elif "たんぱく質" in col:
        user_input[col] = st.slider(col, min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    elif "合計負荷量" in col:
        user_input[col] = st.slider(col, min_value=0.0, max_value=10000.0, value=5000.0, step=10.0)
    else:
        user_input[col] = st.slider(col, min_value=0.0, max_value=500.0, value=100.0, step=1.0)


input_df = pd.DataFrame([user_input])

if st.button("予測する"):
    prediction = clf_model.predict(input_df)[0]
    st.success(f"予測結果：{prediction}")

