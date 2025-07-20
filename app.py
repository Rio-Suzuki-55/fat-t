import streamlit as st
import pandas as pd
import joblib

st.markdown("# 体脂肪率変化予測アプリ \n## 増加・維持・減少 判定+-0.25%ver")

try:
    clf_model = joblib.load("clf_model.pkl")
    input_columns = joblib.load("input_columns.pkl")
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

st.write("以下の入力から、翌日の体脂肪率の変化を予測します。")

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

    # === 🔍 予測確率の可視化 ===
    try:
        proba = clf_model.predict_proba(input_df)[0]
        classes = clf_model.classes_

        proba_df = pd.DataFrame({
            'クラス': classes,
            '予測確率': proba
        }).sort_values(by='予測確率', ascending=False)

        st.subheader("📊 クラスごとの予測確率")
        st.bar_chart(proba_df.set_index('クラス'))
    except AttributeError:
        st.warning("このモデルは確率予測（predict_proba）に対応していません。")


