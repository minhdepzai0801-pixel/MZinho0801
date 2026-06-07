# 🇻🇳 VN AIDEOM-VN

**AI-Driven Decision Optimization Model for Vietnam**
Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI.

Dashboard Streamlit mô phỏng **12 bài toán ra quyết định** trên dữ liệu thực tế của Việt Nam giai đoạn 2020–2025, kết hợp tối ưu hóa (LP, MIP, đa mục tiêu, động, ngẫu nhiên) và học tăng cường.

> **Bài tập lớn:** Các mô hình ra quyết định
> **Sinh viên:** Vũ Công Minh — **MSV:** 23051329
> Trường Đại học Kinh tế, ĐHQGHN — Viện Quản trị kinh doanh

---

## 📌 Tính năng

- **Trang chủ** — tổng quan kinh tế Việt Nam 2024–2025 và 3 bộ dữ liệu gốc (vĩ mô / 10 ngành / 6 vùng).
- **12 bài tập**, mỗi bài trình bày theo logic: **Bối cảnh → Mô hình → Dữ liệu → Tính toán → Chính sách**.
- **Tham số tương tác** riêng cho từng bài ở thanh bên (sidebar).
- **Bài 12** tích hợp 6 module (M1–M6) thành hệ thống AIDEOM-VN hoàn chỉnh.

---

## 🗂️ Danh mục 12 bài

| Cấp độ | Bài | Nội dung | Kỹ thuật chính |
|--------|-----|----------|----------------|
| **Dễ** | Bài 1 | Hàm sản xuất Cobb-Douglas mở rộng (AI, số hóa) | `numpy`, `pandas` |
| | Bài 2 | LP phân bổ ngân sách 4 hạng mục đầu tư số | `scipy.optimize`, `pulp` |
| | Bài 3 | Chỉ số ưu tiên ngành Priorityᵢ (10 ngành) | `numpy`, `pandas` |
| **Trung bình** | Bài 4 | LP phân bổ ngân sách ngành-vùng (công bằng vùng miền) | `pulp`, `cvxpy` |
| | Bài 5 | MIP lựa chọn 15 dự án chuyển đổi số | `pulp` (CBC) |
| | Bài 6 | TOPSIS xếp hạng 6 vùng (Expert / Entropy / AHP) | `numpy` |
| **Khá khó** | Bài 7 | NSGA-II tối ưu 4 mục tiêu Pareto | `pymoo` |
| | Bài 8 | Tối ưu động liên thời gian 2026–2035 | `scipy.optimize` (SLSQP) |
| | Bài 9 | Tác động AI tới lao động (NetJob ròng) | `scipy.optimize` |
| **Khó** | Bài 10 | Quy hoạch ngẫu nhiên 2 giai đoạn (VSS, EVPI, robust) | `pyomo` + HiGHS/GLPK |
| | Bài 11 | Q-learning chính sách kinh tế thích nghi (MDP 81 trạng thái) | `numpy` |
| | Bài 12 | Đồ án tích hợp AIDEOM-VN: 6 module, 5 kịch bản | `streamlit`, `matplotlib` |

---

## 🚀 Cài đặt và chạy

Yêu cầu: **Python 3.10+**

```bash
# 1. (Khuyến nghị) tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Chạy dashboard
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`.

> ⚠️ Đặt 3 tệp dữ liệu CSV cùng thư mục với `app.py` (hoặc trong thư mục `data/`).
> Bài 10 (Pyomo) cần solver: cài `pip install highspy` — nếu không có solver, app tự hiển thị kết quả tham chiếu.

---

## 📁 Cấu trúc thư mục

```
.
├── app.py                          # Ứng dụng Streamlit (toàn bộ 12 bài)
├── requirements.txt                # Danh sách thư viện
├── README.md                       # File này
├── vietnam_macro_2020_2025.csv     # Dữ liệu vĩ mô 2020–2025
├── vietnam_sectors_2024.csv        # Dữ liệu 10 ngành 2024
└── vietnam_regions_2024.csv        # Dữ liệu 6 vùng KT-XH 2024
```

---

## 📊 Nguồn dữ liệu

Cục Thống kê quốc gia (NSO/GSO), Bộ Khoa học và Công nghệ (MoST), Bộ Thông tin và Truyền thông (MIC),
Bộ Kế hoạch và Đầu tư (MPI), World Bank, và báo cáo Global Innovation Index 2025 (WIPO).

*Lưu ý: số liệu trong CSV được làm tròn phục vụ mục đích học tập.*

---

## 📦 Thư viện sử dụng

`streamlit` · `numpy` · `pandas` · `scipy` · `matplotlib` · `pulp` · `pymoo` · `pyomo` · `highspy`

---

## 📄 Giấy phép

Dự án phục vụ mục đích học tập (bài tập lớn học phần Các mô hình ra quyết định).
