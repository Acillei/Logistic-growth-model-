import csv
import json
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. TẢI DỮ LIỆU THỰC TỪ WORLD BANK OPEN DATA (MÃ CHỈ SỐ: SP.POP.TOTL)
def fetch_worldbank_vietnam_pop(csv_filename="vietnam_population_1960_2024.csv"):
    url = (
        "https://api.worldbank.org/v2/country/VNM/indicator/SP.POP.TOTL"
        "?date=1960:2024&format=json&per_page=100"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    records = []
    for entry in payload[1]:
        if entry["value"] is not None:
            year = int(entry["date"])
            pop_raw = int(entry["value"])
            pop_millions = pop_raw / 1e6
            records.append((year, pop_raw, pop_millions))

    records.sort(key=lambda x: x[0])

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Year", "Population_Total", "Population_Millions"])
        writer.writerows(records)

    with open("METADATA_Vietnam_Population.txt", mode="w", encoding="utf-8") as f:
        f.write("TÊN BỘ DỮ LIỆU: Dân số tổng cộng Việt Nam (1960 - 2024)\n")
        f.write("NGUỒN GỐC: World Bank Open Data (Indicator ID: SP.POP.TOTL, Country: VNM)\n")
        f.write("TRUY XUẤT TỪ: https://data.worldbank.org/indicator/SP.POP.TOTL?locations=VN\n")
        f.write("GIẤY PHÉP SỬ DỤNG: Creative Commons Attribution 4.0 International (CC BY 4.0)\n")
        f.write("ĐƠN VỊ ĐO: Người (Population_Total) và Triệu người (Population_Millions)\n")
        f.write(f"SỐ QUAN TRẮC SẠCH: {len(records)} năm liên tục ({records[0][0]} - {records[-1][0]})\n")

    years = np.array([r[0] for r in records], dtype=float)
    pop_m = np.array([r[2] for r in records], dtype=float)
    return years, pop_m


# 2. MÔ HÌNH TOÁN HỌC & MÔ HÌNH TÍNH TOÁN (TỰ CÀI ĐẶT RK4 & EULER TỪ ĐẦU)
def logistic_ode(t, N, r, K):
    """Phương trình vi phân thường Logistic: dN/dt = r * N * (1 - N / K)."""
    return r * N * (1.0 - N / K)


def logistic_analytic(t, r, K, N0, t0=1960.0):
    """Nghiệm giải tích chính xác dùng để hiệu chỉnh tham số và kiểm tra sai số số học."""
    return K / (1.0 + ((K - N0) / N0) * np.exp(-r * (t - t0)))


def solve_rk4(t0, t_end, N0, dt, r, K):
    """Giải số phương trình Logistic bằng thuật toán Runge-Kutta bậc 4 (RK4)."""
    n_steps = int(round((t_end - t0) / dt))
    t = np.linspace(t0, t_end, n_steps + 1)
    N = np.zeros(n_steps + 1)
    N[0] = N0
    for i in range(n_steps):
        k1 = logistic_ode(t[i], N[i], r, K)
        k2 = logistic_ode(t[i] + 0.5 * dt, N[i] + 0.5 * dt * k1, r, K)
        k3 = logistic_ode(t[i] + 0.5 * dt, N[i] + 0.5 * dt * k2, r, K)
        k4 = logistic_ode(t[i] + dt, N[i] + dt * k3, r, K)
        N[i + 1] = N[i] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return t, N


def solve_euler(t0, t_end, N0, dt, r, K):
    """Giải số bằng phương pháp Euler hiện để đối chiếu sai số bước lưới."""
    n_steps = int(round((t_end - t0) / dt))
    t = np.linspace(t0, t_end, n_steps + 1)
    N = np.zeros(n_steps + 1)
    N[0] = N0
    for i in range(n_steps):
        N[i + 1] = N[i] + dt * logistic_ode(t[i], N[i], r, K)
    return t, N


# 3. HIỆU CHỈNH THAM SỐ, ĐÁNH GIÁ ĐỘ KHỚP & BOOTSTRAP DẢI TIN CẬY 95%
def compute_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    return rmse, mae, r2, residuals


def main():
    years, pop_m = fetch_worldbank_vietnam_pop()
    t0 = years[0]
    N0 = pop_m[0]

    popt, pcov = curve_fit(
        lambda t, r, K: logistic_analytic(t, r, K, N0=N0, t0=t0),
        years,
        pop_m,
        p0=[0.035, 130.0],
        bounds=([0.001, 50.0], [0.5, 500.0]),
    )
    r_hat, K_hat = popt
    perr = np.sqrt(np.diag(pcov))

    t_rk4, N_rk4 = solve_rk4(t0, years[-1], N0, dt=0.1, r=r_hat, K=K_hat)
    N_pred_obs = np.interp(years, t_rk4, N_rk4)
    rmse, mae, r2, residuals = compute_metrics(pop_m, N_pred_obs)

    t_inflection = t0 + (1.0 / r_hat) * np.log((K_hat - N0) / N0)
    N_inflection = K_hat / 2.0

    N_exact_obs = logistic_analytic(years, r_hat, K_hat, N0=N0, t0=t0)
    _, N_eu_1 = solve_euler(t0, years[-1], N0, dt=1.0, r=r_hat, K=K_hat)
    _, N_rk_1 = solve_rk4(t0, years[-1], N0, dt=1.0, r=r_hat, K=K_hat)
    err_euler_1 = np.max(np.abs(N_exact_obs - N_eu_1))
    err_rk4_1 = np.max(np.abs(N_exact_obs - N_rk_1))
    err_rk4_01 = np.max(np.abs(N_exact_obs - N_pred_obs))

    print("=== KẾT QUẢ HIỆU CHỈNH MÔ HÌNH LOGISTIC (DÂN SỐ VIỆT NAM 1960-2024) ===")
    print(f"Điều kiện đầu N0 (năm {int(t0)}): {N0:.4f} triệu người")
    print(f"Tốc độ tăng trưởng nội tại r : {r_hat:.6f} ± {perr[0]:.6f} (năm^-1)")
    print(f"Sức chứa môi trường K       : {K_hat:.4f} ± {perr[1]:.4f} (triệu người)")
    print(f"Điểm uốn (t*, K/2)          : Năm {t_inflection:.2f} | Dân số {N_inflection:.2f} triệu người")
    print(f"Độ khớp RMSE                : {rmse:.4f} triệu người")
    print(f"Độ khớp MAE                 : {mae:.4f} triệu người")
    print(f"Hệ số xác định R^2          : {r2:.6f}")
    print(f"Sai số số học tối đa (dt=1.0): Euler = {err_euler_1:.6f} | RK4 = {err_rk4_1:.6e}")
    print(f"Sai số số học tối đa (dt=0.1): RK4 = {err_rk4_01:.6e}")

    # 4. BOOTSTRAP 1.000 LẦN TẠO DẢI TIN CẬY 95% & DỰ BÁO ĐẾN NĂM 2060
    rng = np.random.default_rng(42)
    t_future = np.linspace(t0, 2060.0, 1001)
    boot_trajectories = []
    boot_r, boot_K = [], []

    for _ in range(1000):
        resampled_res = rng.choice(residuals, size=len(residuals), replace=True)
        y_boot = N_pred_obs + resampled_res
        try:
            p_b, _ = curve_fit(
                lambda t, r, K: logistic_analytic(t, r, K, N0=N0, t0=t0),
                years,
                y_boot,
                p0=[r_hat, K_hat],
                bounds=([0.001, 50.0], [0.5, 500.0]),
                maxfev=5000,
            )
            boot_r.append(p_b[0])
            boot_K.append(p_b[1])
            boot_trajectories.append(logistic_analytic(t_future, p_b[0], p_b[1], N0=N0, t0=t0))
        except RuntimeError:
            continue

    boot_trajectories = np.array(boot_trajectories)
    ci_lower = np.percentile(boot_trajectories, 2.5, axis=0)
    ci_upper = np.percentile(boot_trajectories, 97.5, axis=0)
    N_future_mean = logistic_analytic(t_future, r_hat, K_hat, N0=N0, t0=t0)

    print(f"Khoảng tin cậy 95% của r    : [{np.percentile(boot_r, 2.5):.6f}, {np.percentile(boot_r, 97.5):.6f}]")
    print(f"Khoảng tin cậy 95% của K    : [{np.percentile(boot_K, 2.5):.2f}, {np.percentile(boot_K, 97.5):.2f}] triệu người")

    # VẼ BIỂU ĐỒ 1: DỮ LIỆU THỰC, ĐƯỜNG KHỚP LOGISTIC VÀ DẢI TIN CẬY 95%
    plt.figure(figsize=(10, 5.5), dpi=300)
    plt.scatter(years, pop_m, color="#1f77b4", s=22, alpha=0.85, label="Dữ liệu thực (World Bank 1960–2024)")
    plt.plot(t_future, N_future_mean, color="#d62728", linewidth=2.2, label=f"Mô hình Logistic RK4 (R² = {r2:.4f})")
    plt.fill_between(t_future, ci_lower, ci_upper, color="#d62728", alpha=0.2, label="Dải tin cậy 95% (Bootstrap n=1000)")
    plt.axhline(K_hat, color="#2ca02c", linestyle="--", linewidth=1.3, label=f"Ngưỡng sức chứa K = {K_hat:.2f} triệu")
    plt.scatter([t_inflection], [N_inflection], color="#ff7f0e", zorder=5, s=60, label=f"Điểm uốn ({t_inflection:.1f}, {N_inflection:.1f}M)")
    plt.title("Mô hình hóa & Dự báo Tăng trưởng Dân số Việt Nam theo Phương trình Logistic", fontsize=12, fontweight="bold")
    plt.xlabel("Năm", fontsize=11)
    plt.ylabel("Dân số (Triệu người)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("fig1_logistic_fit_ci.png")
    plt.close()

    # 5. PHÂN TÍCH ĐỘ NHẠY 2 THAM SỐ (r VÀ K BIẾN THIÊN ±10%, ±20%)
    factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    labels = ["-20%", "-10%", "Chuẩn (0%)", "+10%", "+20%"]
    colors = ["#440154", "#31688e", "#d62728", "#35b779", "#fde725"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300, sharey=True)

    print("\n=== BẢNG PHÂN TÍCH ĐỘ NHẠY THAM SỐ r VÀ K ===")
    for f_val, lbl, col in zip(factors, labels, colors):
        r_test = r_hat * f_val
        t_s, N_s = solve_rk4(t0, 2060.0, N0, dt=0.1, r=r_test, K=K_hat)
        N_s_obs = np.interp(years, t_s, N_s)
        rmse_r, _, _, _ = compute_metrics(pop_m, N_s_obs)
        t_inf_r = t0 + (1.0 / r_test) * np.log((K_hat - N0) / N0)
        print(f"[Độ nhạy r {lbl:>9}] r = {r_test:.5f} | Điểm uốn t* = {t_inf_r:.1f} | Dân số 2050 = {np.interp(2050, t_s, N_s):.2f}M | RMSE = {rmse_r:.2f}")
        lw = 2.4 if f_val == 1.0 else 1.5
        ls = "-" if f_val == 1.0 else "--"
        axes[0].plot(t_s, N_s, color=col, linewidth=lw, linestyle=ls, label=f"r {lbl} ({r_test:.4f})")

    axes[0].scatter(years, pop_m, color="black", s=10, alpha=0.4, label="Dữ liệu thực")
    axes[0].set_title("Độ nhạy theo tốc độ tăng trưởng nội tại (r)", fontweight="bold")
    axes[0].set_xlabel("Năm")
    axes[0].set_ylabel("Dân số (Triệu người)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    for f_val, lbl, col in zip(factors, labels, colors):
        K_test = K_hat * f_val
        t_s, N_s = solve_rk4(t0, 2060.0, N0, dt=0.1, r=r_hat, K=K_test)
        N_s_obs = np.interp(years, t_s, N_s)
        rmse_K, _, _, _ = compute_metrics(pop_m, N_s_obs)
        t_inf_K = t0 + (1.0 / r_hat) * np.log((K_test - N0) / N0)
        print(f"[Độ nhạy K {lbl:>9}] K = {K_test:.2f}M | Điểm uốn t* = {t_inf_K:.1f} | Dân số 2050 = {np.interp(2050, t_s, N_s):.2f}M | RMSE = {rmse_K:.2f}")
        lw = 2.4 if f_val == 1.0 else 1.5
        ls = "-" if f_val == 1.0 else "--"
        axes[1].plot(t_s, N_s, color=col, linewidth=lw, linestyle=ls, label=f"K {lbl} ({K_test:.1f}M)")

    axes[1].scatter(years, pop_m, color="black", s=10, alpha=0.4, label="Dữ liệu thực")
    axes[1].set_title("Độ nhạy theo sức chứa môi trường (K)", fontweight="bold")
    axes[1].set_xlabel("Năm")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("fig2_sensitivity_analysis.png")
    plt.close()

    # 6. BIỂU ĐỒ PHẦN DƯ (RESIDUALS) ĐỂ SOI VÙNG MÔ HÌNH SAI LỆCH
    plt.figure(figsize=(10, 4), dpi=300)
    plt.bar(years, residuals, color=np.where(residuals >= 0, "#2b8cbe", "#e34a33"), width=0.7)
    plt.axhline(0, color="black", linewidth=1.0)
    plt.title("Phân tích Phần dư (Dữ liệu thực − Mô hình Logistic) giai đoạn 1960–2024", fontsize=11, fontweight="bold")
    plt.xlabel("Năm")
    plt.ylabel("Sai số phần dư (Triệu người)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("fig3_residuals_over_time.png")
    plt.close()


if __name__ == "__main__":
    main()