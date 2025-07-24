import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 支援繁體中文
matplotlib.rcParams['axes.unicode_minus'] = False

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

# --- 統計類別數量 ---
incorrect_mask = df_day["mask_status"].value_counts().get("Incorrect_Mask", 0)
mask = df_day["mask_status"].value_counts().get("Mask", 0)
no_mask = df_day["mask_status"].value_counts().get("No_Mask", 0)
total = incorrect_mask + mask + no_mask

# --- 數字顯示 ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("總人數", total)
col2.metric("⚠️ 配戴不正確", incorrect_mask)
col3.metric("✅ 有戴口罩", mask)
col4.metric("❌ 沒戴口罩", no_mask)

# --- 圓餅圖 ---
st.subheader("🧭 配戴狀況比例")

if total == 0:
    st.warning("⚠️ 今日無任何配戴記錄，無法產生圓餅圖。")
else:
    fig1, ax1 = plt.subplots()
    ax1.pie(
        [incorrect_mask, mask, no_mask],
        labels=["配戴不正確", "有戴口罩", "沒戴口罩"],
        autopct="%1.1f%%",
        colors=["#ffa726", "#66bb6a", "#ef5350"]
    )
    st.pyplot(fig1)

# --- 每小時分布 ---
st.subheader("🕒 每小時配戴分布")
hourly_stats = df_day.groupby(["hour", "mask_status"]).size().unstack().fillna(0)

# ✅ 僅選取存在的欄位（避免 KeyError）
expected_columns = ["Incorrect_Mask", "Mask", "No_Mask"]
available_columns = [col for col in expected_columns if col in hourly_stats.columns]
hourly_stats = hourly_stats[available_columns]

# --- 繪製長條圖
fig2, ax2 = plt.subplots()
hourly_stats.plot(kind="bar", stacked=True, ax=ax2,
                  color=["#ffa726", "#66bb6a", "#ef5350"][:len(available_columns)])
ax2.set_xlabel("小時")
ax2.set_ylabel("人數")
ax2.set_title("每小時配戴狀況")
st.pyplot(fig2)

# --- 詳細資料表 ---
st.subheader("📄 詳細記錄")
st.dataframe(df_day)

# --- 下載按鈕 ---
csv = df_day.to_csv(index=False).encode('utf-8')
st.download_button("💾 下載今日紀錄", csv,
                   file_name=f"{selected_date}_mask_log.csv",
                   mime="text/csv")
