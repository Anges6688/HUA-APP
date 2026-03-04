import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import datetime as dt

# ============ 1. 加载 XGBoost 模型（只保留 XGB） ============
with open("2.训练集构建模型/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# ============ 2. 页面基础设置 ============
st.set_page_config(page_title="儿童高尿酸风险评估工具", layout="centered")

st.title("儿童高尿酸血症风险评估（XGBoost 模型）")

st.markdown("""
**重要说明**

- 本工具基于单中心住院患儿数据集开发，目前主要用于**科研分析与方法学探索**。
- 在体检或基层门诊场景下，可作为**高尿酸血症风险初筛与分层管理的辅助工具**。
- 工具输出结果**不能单独作为诊断依据或处方依据**，任何诊疗决策仍需由有资质的临床医生综合判断。
""")

# ============ 3. 输入模块：性别 + 血压 + 身高体重 ============
st.subheader("1. 基本信息与体格指标")

col1, col2 = st.columns(2)
with col1:
    gender_str = st.radio("性别", ["男孩", "女孩"])
with col2:
    sbp = st.number_input("收缩压 SBP (mmHg)", min_value=60.0, max_value=200.0, value=110.0, step=1.0)

col3, col4 = st.columns(2)
with col3:
    height = st.number_input("身高 (cm)", min_value=40.0, max_value=220.0, value=130.0, step=0.1)
with col4:
    weight = st.number_input("体重 (kg)", min_value=3.0, max_value=200.0, value=30.0, step=0.1)

# ============ 4. 输入模块：出生日期 → 自动计算月龄 ============
st.subheader("2. 年龄信息")

col5, col6 = st.columns(2)
with col5:
    birth_date = st.date_input(
        "出生日期",
        min_value=dt.date(2000, 1, 1),
        max_value=dt.date(2030, 12, 31)
    )
with col6:
    meas_date = st.date_input("测量 / 就诊日期", value=dt.date.today())

# ============ 5. 特征自动计算：Gender, Age_months, BMI ============
# 性别编码：和你建模时保持一致（假设：男=1，女=0）
gender = 1 if gender_str == "男孩" else 0

# 月龄（按天差 / 30.4375 近似换算为月）
days = (meas_date - birth_date).days
age_months = max(days / 30.4375, 0)

# 体质指数 BMI
height_m = height / 100.0
bmi = weight / (height_m ** 2) if height_m > 0 else 0

st.markdown(
    f"**自动计算结果：** 月龄约为 `{age_months:.1f}` 月；体质指数 BMI = `{bmi:.1f}` kg/m²"
)

# ============ 6. 调用 XGBoost 模型进行预测 ============
st.subheader("3. 风险预测")

if st.button("计算高尿酸血症风险（SUA ≥ 420 μmol/L）"):
    # 构建与训练时一致的特征矩阵
    X_input = pd.DataFrame([{
        "Gender": gender,
        "SBP": sbp,
        "Age_months": age_months,
        "Body_Mass_Index": bmi
    }])

    try:
        # 预测概率（正类 = HUA）
        proba = xgb_model.predict_proba(X_input)[0, 1]
        risk_pct = proba * 100

        st.success(f"预测患儿发生高尿酸血症的概率约为：**{risk_pct:.1f}%**")

        # 简单风险分层（阈值你后续可以根据实际再微调）
        if risk_pct < 10:
            level = "低风险"
            note = "目前预测风险较低，可结合体检结果和家族史常规随访。"
        elif risk_pct < 25:
            level = "中低风险"
            note = "建议关注体重管理与血压控制，必要时随访复查 SUA。"
        elif risk_pct < 50:
            level = "中等风险"
            note = "建议结合临床情况评估是否提前安排实验室检测和生活方式干预。"
        else:
            level = "中高或以上风险"
            note = "建议尽快完善血生化检查（含 SUA、肾功能等），并由专科医生评估。"

        st.info(f"风险分层：**{level}**\n\n{note}")

        st.caption(
            "提示：该风险评估结果基于模型训练数据的统计关联，适合作为体检或基层门诊的辅助筛查工具，"
            "不能替代专业医生的临床判断。"
        )

    except Exception as e:
        st.error("预测时出现错误，请检查输入或联系开发者（可能是模型文件路径或特征顺序问题）。")
        st.exception(e)