import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_vn_model import solve_rk4, solve_euler, compute_metrics

st.set_page_config(page_title="Mô phỏng Logistic - Dân số Việt Nam", layout="wide")
st.title("📈 Mô phỏng Mô hình Tăng trưởng Quần thể Logistic – Dân số Việt Nam")

df = pd.read_csv("vietnam_population_1960_2024.csv")
years = df["Year"].values.astype(float)
pop_m = df["Population_Millions"].values.astype(float)

st.sidebar.header("🎛️ Bảng điều khiển Tham số")
r = st.sidebar.slider("Tốc độ tăng trưởng nội tại (r)", min_value=0.010, max_value=0.080, value=0.036, step=0.001, format="%.3f")
K = st.sidebar.slider("Sức chứa môi trường K (triệu người)", min_value=80.0, max_value=220.0, value=135.0, step=1.0)
N0 = st.sidebar.slider("Dân số ban đầu N0 năm 1960 (triệu người)", min_value=20.0, max_value=45.0, value=float(round(pop_m[0], 2)), step=0.5)
dt = st.sidebar.select_slider("Bước lưới thời gian Δt (năm)", options=[5.0, 2.0, 1.0, 0.5, 0.1], value=0.1)
show_euler = st.sidebar.checkbox("So sánh thêm phương pháp Euler hiện", value=False)

t_sim, N_sim = solve_rk4(years[0], 2060.0, N0, dt=dt, r=r, K=K)
N_obs_pred = np.interp(years, t_sim, N_sim)
rmse, mae, r2, _ = compute_metrics(pop_m, N_obs_pred)
t_inf = years[0] + (1.0 / r) * np.log((K - N0) / N0) if K > N0 else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sai số RMSE", f"{rmse:.3f} triệu")
col2.metric("Hệ số khớp R²", f"{r2:.4f}")
col3.metric("Năm đạt điểm uốn (K/2)", f"{t_inf:.1f}" if not np.isnan(t_inf) else "N/A")
col4.metric("Dự báo năm 2050", f"{np.interp(2050.0, t_sim, N_sim):.2f} triệu")

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.scatter(years, pop_m, color="#1f77b4", s=25, label="Dữ liệu thực World Bank (1960–2024)")
ax.plot(t_sim, N_sim, color="#d62728", linewidth=2.3, label=f"Mô phỏng RK4 (Δt = {dt})")
if show_euler:
    t_eu, N_eu = solve_euler(years[0], 2060.0, N0, dt=dt, r=r, K=K)
    ax.plot(t_eu, N_eu, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"Mô phỏng Euler (Δt = {dt})")
ax.axhline(K, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"Ngưỡng bão hòa K = {K:.1f} triệu")
ax.set_xlabel("Năm")
ax.set_ylabel("Dân số (Triệu người)")
ax.set_title("So sánh Quỹ đạo Mô phỏng Logistic với Dữ liệu Thực tế Việt Nam")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()
st.pyplot(fig)