import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 加载模型和编码器
rf = joblib.load('rf_model.pkl')
le = joblib.load('label_encoder.pkl')

def optimize_ab(fabric_name, temp, humidity, model):
    best_logR = np.inf
    best_A, best_B = 0, 0
    fabric_encoded = le.transform([fabric_name])[0]
    for A in range(0, 15):
        for B in range(0, 15):
            total = A + B
            if total < 4 or total > 14:
                continue
            x = np.array([[fabric_encoded, A, B, temp, humidity]])
            pred_logR = model.predict(x)[0]
            if pred_logR < best_logR:
                best_logR = pred_logR
                best_A, best_B = A, B
    best_R = 10 ** best_logR
    return best_A, best_B, best_R, best_logR

st.set_page_config(page_title="抗静电助剂配比推荐", layout="centered")
st.title("🧪 抗静电性能优化助手")
st.markdown("根据布料、温度、湿度，推荐A、B助剂的最佳浓度配比（总浓度4~14 wt%）")

col1, col2 = st.columns(2)
with col1:
    fabric = st.selectbox("布料类型", ['棉', '涤纶', '涤棉65/35'])
    temp = st.slider("温度 (℃)", 20, 30, 25, step=1)
with col2:
    humidity = st.slider("相对湿度 (%)", 30, 70, 50, step=1)

if st.button("🔍 推荐配比", type="primary"):
    with st.spinner("计算中..."):
        A_opt, B_opt, R_opt, logR_opt = optimize_ab(fabric, temp, humidity, rf)
    st.success("✅ 推荐结果")
    st.metric("最佳 A 浓度 (wt%)", f"{A_opt} %")
    st.metric("最佳 B 浓度 (wt%)", f"{B_opt} %")
    st.metric("总浓度", f"{A_opt+B_opt} %")
    st.metric("预测表面电阻率", f"{R_opt:.2e} Ω/sq", help=f"log10值 = {logR_opt:.3f}")

    # 附近配比参考
    st.markdown("#### 📊 附近配比参考（总浓度相同）")
    candidates = []
    for delta in [-1, 0, 1]:
        A_try = A_opt + delta
        B_try = B_opt - delta
        if 0 <= A_try <= 14 and 0 <= B_try <= 14 and 4 <= A_try+B_try <= 14:
            x = np.array([[le.transform([fabric])[0], A_try, B_try, temp, humidity]])
            logR_pred = rf.predict(x)[0]
            candidates.append((A_try, B_try, 10**logR_pred))
    if candidates:
        cand_df = pd.DataFrame(candidates, columns=['A%', 'B%', '电阻率(Ω/sq)'])
        cand_df['电阻率(Ω/sq)'] = cand_df['电阻率(Ω/sq)'].apply(lambda x: f"{x:.2e}")
        st.dataframe(cand_df, use_container_width=True)

st.markdown("---")
st.caption("模型基于随机森林，训练数据涵盖总浓度4-14%，温度20-30℃，湿度30-70%。")
