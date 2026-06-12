# 🎨 AIDEOM-VN — Bản giao diện "Decision Blueprint" (app_neo.py)

Đây là **bản Streamlit giao diện mới** của dự án AIDEOM-VN, dùng chung dữ liệu và
**giữ nguyên 100% kết quả tính toán** với `app.py`, chỉ khác phần trình bày:

| | `app.py` (bản gốc) | `app_neo.py` (bản mới) |
|---|---|---|
| Giao diện | Sáng, sidebar trái | **Tối "midnight blueprint"**, lưới kỹ thuật |
| Điều hướng | Radio dọc ở sidebar | **Topbar ngang trên đỉnh** |
| Phông chữ | Mặc định | **Space Grotesk + IBM Plex Mono** |
| Màu nhấn | Pastel xanh/tím | **Cyan + Amber** trên nền xanh đen |
| Tham số | Sidebar | Expander "⚙️ Tham số mô hình" dưới topbar |
| Cấu trúc 5 trang | Bối cảnh→Mô hình→Dữ liệu→Tính toán→Chính sách | **Giữ nguyên** |
| Kết quả tính toán | — | **Giữ nguyên y hệt** |

## Chạy

```bash
pip install -r requirements.txt
streamlit run app_neo.py        # bản giao diện mới
# hoặc
streamlit run app.py            # bản gốc
```

3 tệp CSV (`vietnam_macro_2020_2025.csv`, `vietnam_sectors_2024.csv`,
`vietnam_regions_2024.csv`) phải nằm cùng thư mục.

## Deploy Streamlit Cloud
Giống `app.py`, chỉ cần đặt **Main file path = `app_neo.py`** khi tạo app trên
share.streamlit.io. Có thể deploy cả hai làm hai app riêng từ cùng một repo.

