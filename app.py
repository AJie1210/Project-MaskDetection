import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 顯示繁體中文
matplotlib.rcParams['axes.unicode_minus'] = False          # 顯示負號

st.set_page_config(page_title="口罩配戴統計儀表板", layout="wide")
st.title("😷 口罩配戴追蹤與統計儀表板")

# --- 載入資料 ---
@st.cache_data
def load_data():
    df = pd.read_csv("tracking_mask_log.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df

df = load_data()

# --- 日期篩選 ---
selected_date = st.sidebar.date_input("📅 選擇日期", value=pd.Timestamp.today().date())
df_day = df[df["date"] == selected_date]

# --- 基本統計 ---
total = len(df_day)
masked = (df_day["mask_status"] == "masked").sum()
no_mask = (df_day["mask_status"] == "no_mask").sum()
incorrect = (df_day["mask_status"] == "incorrect_mask").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 總人數", total)
col2.metric("✅ 正確配戴", masked)
col3.metric("❌ 沒戴口罩", no_mask)
col4.metric("⚠️ 配戴不正確", incorrect)

# --- 圓餅圖 ---
st.subheader("🧭 配戴狀況比例")
fig1, ax1 = plt.subplots()
ax1.pie(
    [masked, no_mask, incorrect],
    labels=["正確配戴", "沒戴口罩", "配戴不正確"],
    autopct="%1.1f%%",
    colors=["#66bb6a", "#ef5350", "#ffa726"]
)
st.pyplot(fig1)

# --- 每小時統計圖（長條圖）---
st.subheader("🕒 每小時配戴狀況")
hourly_stats = df_day.groupby(["hour", "mask_status"]).size().unstack().fillna(0)
hourly_stats = hourly_stats[["masked", "no_mask", "incorrect_mask"]]  # 排序

fig2, ax2 = plt.subplots()
hourly_stats.plot(kind="bar", stacked=True, ax=ax2,
                  color=["#66bb6a", "#ef5350", "#ffa726"])
ax2.set_xlabel("小時")
ax2.set_ylabel("人數")
ax2.set_title("每小時配戴分布")
st.pyplot(fig2)

# --- 詳細資料表 ---
st.subheader("📄 詳細記錄")
st.dataframe(df_day)

# --- 下載按鈕 ---
csv = df_day.to_csv(index=False).encode('utf-8')
st.download_button("💾 下載今日紀錄", csv,
                   file_name=f"{selected_date}_mask_log.csv",
                   mime="text/csv")
