# Mô Hình Hóa & Mô Phỏng Tăng Trưởng Quần Thể Logistic – Dân Số Việt Nam (1960–2024)

Sản phẩm mã nguồn, dữ liệu thực nghiệm và tài liệu minh chứng cho **Đề 2 – Học phần Phương pháp Nghiên cứu trong Khoa học Tự nhiên** (Trường Đại học Sư phạm – Đại học Đà Nẵng).

* **Sinh viên thực hiện:** Nguyễn Thị Hoàng Anh
* **Mã sinh viên:** 3120225007
* **Lớp học phần:** 25CNTT3
* **Video Demo mô phỏng (3–5 phút):** (https://drive.google.com/drive/folders/1kgCnLolWy-HXLKwLcKAf2NvAV_zzLDgj?usp=drive_link)

---

## Thành phẩm nộp kèm

```text
├── vietnam_population_1960_2024.csv     # Dữ liệu dân số tổng cộng Việt Nam (1960–2024)
├── METADATA_Vietnam_Population.txt      # Mô tả nguồn gốc dữ liệu và giấy phép mở CC BY 4.0
├── logistic_vn_model.py                 # Mã nguồn chính: tải dữ liệu, giải RK4/Euler, hiệu chỉnh, Bootstrap & độ nhạy
├── demo_streamlit.py                    # Ứng dụng giao diện tương tác điều chỉnh trực tiếp tham số r, K, N0, Δt
├── fig1_logistic_fit_ci.png             # Đồ thị đường khớp Logistic kèm dải tin cậy 95% và điểm uốn
├── fig2_sensitivity_analysis.png        # Đồ thị phân tích độ nhạy hai tham số r và K (±10%, ±20%)
├── fig3_residuals_over_time.png         # Đồ thị phân tích phần dư theo từng giai đoạn lịch sử
├── requirements.txt                     # Danh sách thư viện Python cần thiết
└── README.md                            # Tài liệu hướng dẫn sử dụng và mô tả dự án
```

---

## Hướng dẫn cài đặt và sử dụng chi tiết

### Yêu cầu môi trường & Cài đặt thư viện
* **Yêu cầu:** Python phiên bản `3.8` trở lên và kết nối Internet (nếu muốn tải mới dữ liệu trực tiếp từ API World Bank).
* **Bước 1:** Mở Terminal (Command Prompt / PowerShell) tại thư mục gốc của dự án.
* **Bước 2:** Cài đặt toàn bộ thư viện phụ thuộc bằng lệnh:
  ```bash
  python -m pip install -r requirements.txt
  ```

### Sử dụng kịch bản mô phỏng & kiểm định chính (`logistic_vn_model.py`)
1. **Cách chạy chương trình:**
   ```bash
   python logistic_vn_model.py
   ```
2. **Quy trình xử lý tự động của chương trình:**
   * Truy xuất chuỗi dữ liệu dân số Việt Nam giai đoạn 1960–2024 từ API World Bank và xuất ra tệp `vietnam_population_1960_2024.csv` cùng tệp thông tin bản quyền `METADATA_Vietnam_Population.txt`.
   * Hiệu chỉnh cặp tham số tối ưu r, K bằng bình phương tối thiểu phi tuyến (`scipy.optimize.curve_fit`).
   * Giải số phương trình vi phân bằng thuật toán **RK4** và đối chiếu sai số cắt cụt với phương pháp **Euler hiện**.
   * Chạy **Residual Bootstrap (1.000 lần lặp)** để ước lượng khoảng tin cậy 95% và thực hiện phân tích độ nhạy khi biến thiên r, K.
3. **Cách đọc kết quả đầu ra:**
   * **Trên màn hình Terminal:** In đầy đủ các chỉ số r, K, điểm uốn t*, K/2, độ khớp RMSE, MAE, R^2, sai số số học RK4 vs Euler và bảng số liệu độ nhạy chi tiết.
   * **Trong thư mục dự án:** Tự động lưu 3 biểu đồ độ phân giải cao gồm `fig1_logistic_fit_ci.png`, `fig2_sensitivity_analysis.png` và `fig3_residuals_over_time.png`.

### Sử dụng ứng dụng giao diện tương tác (`demo_streamlit.py`)
1. **Cách khởi chạy giao diện web:**
   ```bash
   python -m streamlit run demo_streamlit.py
   ```
   Sau khi chạy lệnh, trình duyệt web sẽ tự động mở tại địa chỉ `http://localhost:8501`. *(Lưu ý: cần có sẵn tệp `vietnam_population_1960_2024.csv` trong cùng thư mục; nếu chưa có, hãy chạy `logistic_vn_model.py` trước).*
2. **Hướng dẫn thao tác trên Bảng điều khiển (Sidebar bên trái):**
   * **Thanh trượt `Tốc độ tăng trưởng nội tại (r)` (`0.010` – `0.080`):** Kéo để quan sát sự thay đổi độ dốc của đường cong chữ S và sự dịch chuyển sớm/muộn của năm đạt điểm uốn.
   * **Thanh trượt `Sức chứa môi trường K` (`80.0` – `220.0` triệu người):** Kéo để thay đổi ngưỡng trần bão hòa dài hạn (đường đứt nét màu xanh lá) và kịch bản dân số năm 2050.
   * **Thanh trượt `Dân số ban đầu N0 năm 1960` (`20.0` – `45.0` triệu người):** Thử nghiệm độ nhạy của quỹ đạo đối với điều kiện đầu.
   * **Thanh chọn `Bước lưới thời gian Δt` (`5.0`, `2.0`, `1.0`, `0.5`, `0.1` năm):** Điều chỉnh kích thước bước tích phân để kiểm tra độ ổn định số học.
   * **Hộp kiểm `So sánh thêm phương pháp Euler hiện`:** Tích chọn để hiển thị thêm đường mô phỏng Euler (màu cam đứt nét); kết hợp kéo di chuyển thanh đo để thấy sự thay đổi.
3. **Cách đọc vùng hiển thị chính:**
   * **4 thẻ chỉ số phía trên:** Cập nhật tức thời sai số `RMSE`, hệ số xác định `R²`, `Năm đạt điểm uốn (K/2)` và `Dự báo năm 2050` ngay khi kéo thanh trượt.
   * **Biểu đồ trung tâm:** Đối chiếu trực quan các điểm dữ liệu thực (chấm tròn xanh dương) với quỹ đạo mô phỏng RK4 (đường liền nét đỏ) và ngưỡng sức chứa K.
