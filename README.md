# 🎨 AIDEOM-VN — Bản giao diện "Decision Blueprint" (app.py)

Dashboard Streamlit mô phỏng *12 bài toán ra quyết định* trên dữ liệu thực tế của
Việt Nam giai đoạn 2020–2025, với giao diện tối "midnight blueprint", điều hướng
bằng thanh ngang trên đỉnh (topbar) và phần tính toán trình bày trực quan.

## Đặc điểm giao diện

- *Giao diện tối "midnight blueprint"* với lưới kỹ thuật, phông chữ Space Grotesk + IBM Plex Mono, màu nhấn cyan + amber trên nền xanh đen.
- *Điều hướng topbar ngang* trên đỉnh; tham số mô hình đặt trong expander "⚙️ Tham số mô hình" ngay dưới topbar.
- *Mỗi bài trình bày theo 5 trang*: Bối cảnh → Mô hình → Dữ liệu → Tính toán → Chính sách.
- *Phần Tính toán trực quan*: hộp các bước tính (worked formula), thẻ kết quả nổi bật, biểu đồ đồng bộ theme và khối nhận định.

## Chạy

pip install -r requirements.txt
streamlit run app.py

3 tệp CSV (vietnam_macro_2020_2025.csv, vietnam_sectors_2024.csv,
vietnam_regions_2024.csv) phải nằm cùng thư mục với app.py.

## Deploy Streamlit Cloud

Đẩy mã nguồn lên GitHub, vào share.streamlit.io → Create app, đặt
**Main file path = app.py**, dùng chung requirements.txt và 3 tệp CSV.
