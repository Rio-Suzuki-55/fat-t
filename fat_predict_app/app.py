import streamlit as st
import pandas as pd
import joblib

st.title("体脂肪率変化予測アプリ")

try:
    clf_model = joblib.load("clf_model.pkl")
    input_columns = joblib.load("input_columns.pkl")
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

st.write("以下の入力から、体脂肪率が【増加・維持・減少】するかを予測します。")

user_input = {}
for col in input_columns:
    if "体重(kg)" in col:
        user_input[col] = st.slider(col, min_value=30.0, max_value=110.0, value=65.0, step=0.1)
    elif "基礎代謝(kcal)" in col:
        user_input[col] = st.slider(col, min_value=1000.0, max_value=3000.0, value=1500.0, step=5.0)
    elif "たんぱく質(g)" in col:
        user_input[col] = st.slider(col, min_value=0.0, max_value=300.0, value=70.0, step=0.1)
    elif "合計負荷量" in col:
        user_input[col] = st.slider(col, min_value=0.0, max_value=10000.0, value=5000.0, step=10.0)
    else:
        user_input[col] = st.slider(col, min_value=0.0, max_value=500.0, value=100.0, step=1.0)

input_df = pd.DataFrame([user_input])

if st.button("予測する"):
    prediction = clf_model.predict(input_df)[0]
    st.success(f"予測結果：{prediction}")



