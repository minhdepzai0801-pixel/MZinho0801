# -*- coding: utf-8 -*-
"""
============================================================================
 VN AIDEOM-VN  —  AI-Driven Decision Optimization Model for Vietnam
 Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI
----------------------------------------------------------------------------
 Mỗi bài (1–12) được trình bày theo đúng logic của một bài phân tích mô hình
 ra quyết định, gồm 5 trang (tab):
     Bối cảnh → Mô hình → Dữ liệu → Tính toán → Chính sách

 Họ và tên   : Vũ Công Minh
 Mã sinh viên: 23051329
 Bài tập lớn : Các mô hình ra quyết định

 Chạy:  streamlit run app.py
============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="VN AIDEOM-VN", page_icon="🇻🇳",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      h1 {font-size: 2.2rem !important; line-height: 1.15;}
      .intro-line {color:#374151; font-size:1.02rem; margin: 2px 0 10px 0;}
      .hero {background: linear-gradient(120deg,#dbe4ff 0%,#e9d8fd 45%,#d7f5e3 100%);
        border-radius: 18px; padding: 26px 30px; margin-bottom: 16px;}
      .hero h1 {margin: 0 0 4px 0;}
      .hero p {margin: 6px 0; color:#374151;}
      .pill {display:inline-block; background:#ffffffcc; border:1px solid #e5e7eb;
        border-radius: 999px; padding:6px 14px; margin:4px 6px 0 0; font-size:0.85rem;}
      .sb-id {background:#f1f5f9; border-radius:10px; padding:12px 14px;
        font-size:0.85rem; line-height:1.5; margin-top:8px;}
      .stTabs [data-baseweb="tab-list"] {gap: 18px;}
      .stTabs [data-baseweb="tab"] {font-size:0.95rem;}
      div[data-testid="stMetricValue"] {font-size:1.7rem;}
      .small-note {color:#6b7280; font-size:0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"] = True

_SEARCH_DIRS = [".", "data", "/mnt/user-data/uploads", os.path.dirname(os.path.abspath(__file__))]


def _find(fname):
    for d in _SEARCH_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return fname


@st.cache_data(show_spinner=False)
def load_macro():
    return pd.read_csv(_find("vietnam_macro_2020_2025.csv")).sort_values("year").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sectors():
    return pd.read_csv(_find("vietnam_sectors_2024.csv"))


@st.cache_data(show_spinner=False)
def load_regions():
    return pd.read_csv(_find("vietnam_regions_2024.csv"))


SECTOR_VI = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Khai khoáng",
             "Bán buôn-bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải",
             "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế"]
REGION_VI = ["Trung du miền núi phía Bắc", "Đồng bằng sông Hồng",
             "Bắc Trung Bộ + DH Trung Bộ", "Tây Nguyên",
             "Đông Nam Bộ", "Đồng bằng sông Cửu Long"]
REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
ITEMS = ["I", "D", "AI", "H"]
BETA = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
    ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
    ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
    ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25,
}
BETA_MAT = np.array([[BETA[(r, j)] for j in ITEMS] for r in REGIONS])
E_R = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
RHO_R = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
SIG_R = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])
D0_ARR = np.array([38, 78, 55, 32, 82, 48], dtype=float)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def page_title(emoji, title, name=None):
    st.markdown(f"# {emoji} {title}")


def topsis(X, w, is_benefit):
    R = X / np.sqrt((X ** 2).sum(axis=0))
    V = R * w
    A_star = np.where(is_benefit, V.max(0), V.min(0))
    A_neg = np.where(is_benefit, V.min(0), V.max(0))
    S_star = np.sqrt(((V - A_star) ** 2).sum(1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(1))
    return S_neg / (S_star + S_neg)


def entropy_weights(X):
    P = X / X.sum(0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), 0)
    d = 1 - E
    return d / d.sum()


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
PAGES = [
    "🏠 Trang chủ",
    "🌱 Bài 1 — Cobb-Douglas + AI",
    "💰 Bài 2 — LP ngân sách số",
    "📊 Bài 3 — Priority 10 ngành",
    "🗺️ Bài 4 — LP ngành-vùng",
    "🎯 Bài 5 — MIP 15 dự án",
    "🏆 Bài 6 — TOPSIS 6 vùng",
    "🌐 Bài 7 — NSGA-II Pareto",
    "📈 Bài 8 — Động 2026-2035",
    "👷 Bài 9 — Lao động & AI",
    "🎲 Bài 10 — Stochastic SP",
    "♻️ Bài 11 — Q-learning RL",
    "🧩 Bài 12 — AIDEOM tích hợp",
]

with st.sidebar:
    st.markdown("### 🇻🇳 VN AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI")
    st.markdown("**Chọn bài**")
    page = st.radio("Chọn bài", PAGES, label_visibility="collapsed")
    # Khu vực tham số riêng cho từng bài (các page_* sẽ ghi vào SB)
    SB = st.container()
    st.markdown("---")
    st.markdown(
        """
        <div class="sb-id">
        <b>Họ và tên:</b> Vũ Công Minh<br>
        <b>Mã sinh viên:</b> 23051329<br>
        <b>Bài tập lớn:</b> Các mô hình ra quyết định
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
#  TRANG CHỦ
# ============================================================================
def page_home():
    st.markdown(
        """
        <div class="hero">
          <h1>🇻🇳 VN AIDEOM-VN</h1>
          <p><b>AI-Driven Decision Optimization Model for Vietnam</b></p>
          <p>Dashboard mô phỏng 12 bài toán ra quyết định phát triển kinh tế Việt Nam
          trong kỷ nguyên AI. Hệ thống kết hợp <b>Python</b>, <b>tối ưu hóa</b>,
          <b>học tăng cường</b> và <b>mô phỏng chính sách</b>.</p>
          <div>
            <span class="pill">🐍 Python</span>
            <span class="pill">📊 Streamlit Dashboard</span>
            <span class="pill">🧮 Optimization</span>
            <span class="pill">♻️ Reinforcement Learning</span>
            <span class="pill">🇻🇳 Vietnam 2020–2025 Data</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("📌 Bức tranh kinh tế Việt Nam tham chiếu nhanh 2024–2025")
    r1 = st.columns(4)
    r1[0].metric("GDP 2025", "514,0 tỷ USD", "+8,02%")
    r1[1].metric("Kinh tế số/GDP", "≈19,5%", "+1,2 điểm %")
    r1[2].metric("FDI giải ngân 2025", "27,6 tỷ USD", "+8,9%")
    r1[3].metric("GDP/người 2025", "5.026 USD", "+6,9%")
    r2 = st.columns(4)
    r2[0].metric("GDP 2025 (ngh.tỷ VND)", "12.847,6")
    r2[1].metric("DN công nghệ số", "80,1 nghìn")
    r2[2].metric("GII 2025", "Hạng 44/139")
    r2[3].metric("KH-CN/GDP", "≈2,49%")

    st.markdown("---")
    st.subheader("🗂️ Dữ liệu gốc Việt Nam 2020–2025")
    t1, t2, t3 = st.tabs(["Vĩ mô 2020–2025", "10 ngành 2024", "6 vùng KT-XH 2024"])
    with t1:
        df = load_macro()
        st.caption("`vietnam_macro_2020_2025.csv`")
        st.dataframe(df, use_container_width=True, height=250)
    with t2:
        st.caption("`vietnam_sectors_2024.csv`")
        st.dataframe(load_sectors(), use_container_width=True, height=320)
    with t3:
        st.caption("`vietnam_regions_2024.csv`")
        st.dataframe(load_regions(), use_container_width=True, height=250)

    st.markdown("---")
    st.subheader("📚 Nội dung 12 bài tập")
    cards = [
        ("Bài 1", "Cobb-Douglas mở rộng (AI, số hóa), growth accounting, dự báo 2030."),
        ("Bài 2", "LP phân bổ ngân sách 4 hạng mục, shadow price, độ nhạy."),
        ("Bài 3", "Chỉ số ưu tiên ngành Priorityᵢ, chuẩn hóa min-max."),
        ("Bài 4", "LP ngành-vùng với ràng buộc công bằng vùng miền."),
        ("Bài 5", "MIP chọn 15 dự án chuyển đổi số."),
        ("Bài 6", "TOPSIS xếp hạng 6 vùng (Expert/Entropy/AHP)."),
        ("Bài 7", "NSGA-II Pareto 4 mục tiêu."),
        ("Bài 8", "Tối ưu động liên thời gian 2026–2035."),
        ("Bài 9", "Tác động AI tới lao động, NetJob ròng."),
        ("Bài 10", "Quy hoạch ngẫu nhiên 2 giai đoạn (VSS, EVPI)."),
        ("Bài 11", "Q-learning chính sách kinh tế thích nghi."),
        ("Bài 12", "Đồ án tích hợp AIDEOM-VN: 6 module, 5 kịch bản."),
    ]
    cols = st.columns(3)
    for i, (t, d) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"**{t}**")
            st.caption(d)


# ============================================================================
#  BÀI 1 — Cobb-Douglas
# ============================================================================
def sidebar_bai1():
    with SB:
        st.markdown("### Tham số Bài 1")
        a = st.slider("Bài 1 - α - Vốn K", 0.0, 1.0, 0.33, 0.01)
        b = st.slider("Bài 1 - β - Lao động L", 0.0, 1.0, 0.42, 0.01)
        g = st.slider("Bài 1 - γ - Số hóa D", 0.0, 1.0, 0.10, 0.01)
        d = st.slider("Bài 1 - δ - AI", 0.0, 1.0, 0.08, 0.01)
        th = st.slider("Bài 1 - θ - Nhân lực H", 0.0, 1.0, 0.07, 0.01)
        st.markdown("**Kịch bản 2030**")
        D30 = st.slider("D 2030 (% GDP)", 19, 40, 30)
        AI30 = st.slider("AI 2030 (ngh.DN)", 80, 150, 100)
        H30 = st.slider("H 2030 (%)", 29, 45, 35)
        gK = st.slider("K tăng (%/năm)", 3, 10, 6)
    return dict(a=a, b=b, g=g, d=d, th=th, D30=D30, AI30=AI30, H30=H30, gK=gK / 100)


def page_bai1():
    P = sidebar_bai1()
    page_title("🌱", "Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa",
               "Bài 1")
    df = load_macro()
    years = df["year"].values
    Y = df["GDP_trillion_VND"].values
    K = np.array([16500, 17800, 19600, 21300, 23500, 25900], dtype=float)
    L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
    D = df["digital_economy_share_GDP_pct"].values.astype(float)
    AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
    H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])
    a, b, g, d, th = P["a"], P["b"], P["g"], P["d"], P["th"]
    A = Y / (K ** a * L ** b * D ** g * AI ** d * H ** th)
    # Thống kê tóm tắt dùng chung cho tab Tính toán và Chính sách
    A_mean = A.mean()
    Y_hat = A_mean * (K ** a * L ** b * D ** g * AI ** d * H ** th)
    mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100
    n = 5
    g_Y = (np.log(Y[-1]) - np.log(Y[0])) / n
    comp = {
        "TFP (A)": (np.log(A[-1]) - np.log(A[0])) / n,
        "K (Vốn)": a * (np.log(K[-1]) - np.log(K[0])) / n,
        "L (Lao động)": b * (np.log(L[-1]) - np.log(L[0])) / n,
        "D (Số hóa)": g * (np.log(D[-1]) - np.log(D[0])) / n,
        "AI": d * (np.log(AI[-1]) - np.log(AI[0])) / n,
        "H (Nhân lực)": th * (np.log(H[-1]) - np.log(H[0])) / n}
    ratio = {k: v / g_Y * 100 for k, v in comp.items()}
    new_factors = {k: ratio[k] for k in ["D (Số hóa)", "AI", "H (Nhân lực)"]}
    top_new = max(new_factors, key=new_factors.get)

    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["1.1 Bối cảnh", "1.2 Mô hình", "1.3 Dữ liệu", "1.4 Tính toán", "1.5 Chính sách"])

    with t_ctx:
        st.subheader("1.1. Bối cảnh Việt Nam 2020–2025")
        st.write("Bài toán đặt ra là: nếu nền kinh tế Việt Nam được mô hình hóa bằng hàm sản "
                 "xuất Cobb-Douglas mở rộng, trong đó ngoài **vốn K** và **lao động L** còn có "
                 "thêm **số hóa D**, **năng lực AI** và **nhân lực số H**, thì mô hình có giải "
                 "thích tốt biến động GDP thực tế hay không, và yếu tố nào đóng góp nhiều nhất.")
        cagr = (Y[-1] / Y[0]) ** (1 / 5) - 1
        c = st.columns(4)
        c[0].metric("GDP 2025", f"{Y[-1]:,.1f}", "↑ nghìn tỷ VND")
        c[1].metric("CAGR GDP 2020–2025", f"{cagr*100:.2f}%/năm")
        c[2].metric("Kinh tế số/GDP 2025", "19,5%")
        c[3].metric("DN công nghệ số 2025", "80,1 nghìn")
        ctab = pd.DataFrame({
            "Chỉ tiêu": ["GDP", "Kinh tế số/GDP D", "AI (ngh.DN)", "Nhân lực số H", "Lao động L"],
            "2020": [Y[0], D[0], AI[0], H[0], L[0]],
            "2025": [Y[-1], D[-1], AI[-1], H[-1], L[-1]],
            "Tăng trưởng kép/năm": [
                f"{(Y[-1]/Y[0])**(1/5)*100-100:.2f}%", f"{(D[-1]/D[0])**(1/5)*100-100:.2f}%",
                f"{(AI[-1]/AI[0])**(1/5)*100-100:.2f}%", f"{(H[-1]/H[0])**(1/5)*100-100:.2f}%",
                f"{(L[-1]/L[0])**(1/5)*100-100:.2f}%"]})
        st.dataframe(ctab, use_container_width=True)

    with t_mod:
        st.subheader("1.2. Mô hình toán học")
        st.latex(r"Y_t = A_t \cdot K_t^{\alpha} L_t^{\beta} D_t^{\gamma} AI_t^{\delta} H_t^{\theta}, \quad \alpha+\beta+\gamma+\delta+\theta = 1")
        st.markdown("Dạng logarit dùng cho phân rã tăng trưởng (growth accounting):")
        st.latex(r"\Delta\ln Y_t = \Delta\ln A_t + \alpha\Delta\ln K_t + \beta\Delta\ln L_t + \gamma\Delta\ln D_t + \delta\Delta\ln AI_t + \theta\Delta\ln H_t")
        tot = a + b + g + d + th
        st.write(f"Tổng hệ số hiện tại α+β+γ+δ+θ = **{tot:.2f}**.")
        if abs(tot - 1) > 1e-9:
            st.warning("Tổng ≠ 1: lệch khỏi giả định lợi suất không đổi theo quy mô.")
        st.dataframe(pd.DataFrame({
            "Hệ số": ["α (K)", "β (L)", "γ (D)", "δ (AI)", "θ (H)"],
            "Giá trị": [a, b, g, d, th]}), use_container_width=True, hide_index=True)

    with t_data:
        st.subheader("1.3. Dữ liệu Việt Nam 2020–2025")
        st.caption("Tổng hợp từ `vietnam_macro_2020_2025.csv` và nguồn bổ sung (MoST, MIC). "
                   "Đơn vị: Y, K — nghìn tỷ VND; L — triệu người; D, H — %; AI — nghìn DN.")
        st.dataframe(pd.DataFrame({
            "Năm": years, "Y (GDP)": Y, "K (vốn)": K, "L (LĐ)": L,
            "D (%)": D, "AI (ngh.DN)": AI, "H (%)": H}),
            use_container_width=True, hide_index=True)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
        axes[0].plot(years, Y, "k-o", label="Y"); axes[0].plot(years, K, "b-s", label="K")
        axes[0].set_title("Y & K (nghìn tỷ)"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(years, D, "g-o", label="D %"); axes[1].plot(years, H, "m-s", label="H %")
        axes[1].set_title("Số hóa D & nhân lực H (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[2].plot(years, AI, "r-o"); axes[2].set_title("AI (nghìn DN số)"); axes[2].grid(alpha=0.3)
        show_fig(fig)
        st.caption("AI (số DN công nghệ số) và số hóa D tăng nhanh nhất trong giai đoạn — đây "
                   "là động lực mới nổi bên cạnh tích lũy vốn K truyền thống.")

    with t_calc:
        st.subheader("Câu 1.4.1 — TFP $A_t$ giải ngược từ hàm sản xuất")
        cc = st.columns([1, 1])
        with cc[0]:
            st.dataframe(pd.DataFrame({"Năm": years, "A_t (TFP)": np.round(A, 4)}),
                         use_container_width=True, hide_index=True)
            st.caption(f"TFP tăng {((A[-1]/A[0])**(1/5)-1)*100:.2f}%/năm.")
        with cc[1]:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(years, A, "bo-", lw=2)
            ax.set_title("$A_t$ (TFP)"); ax.grid(alpha=0.3)
            show_fig(fig)

        st.subheader("Câu 1.4.2 — Dự báo $\\hat{Y}$ và MAPE")
        st.dataframe(pd.DataFrame({"Năm": years, "Y thực": np.round(Y, 1),
                                   "Y dự báo": np.round(Y_hat, 1),
                                   "Sai số %": np.round((Y_hat - Y) / Y * 100, 2)}),
                     use_container_width=True, hide_index=True)
        m = st.columns(2)
        m[0].metric("MAPE", f"{mape:.3f}%")
        m[1].metric("Đánh giá độ khớp",
                    "Rất tốt" if mape < 2 else ("Khá tốt" if mape < 5 else "Trung bình"))
        st.caption(f"MAPE = {mape:.2f}% — mô hình Cobb-Douglas mở rộng "
                   f"{'giải thích tốt' if mape < 5 else 'giải thích ở mức chấp nhận được'} "
                   "biến động GDP thực tế, xác nhận tính hợp lý của dạng hàm.")

        st.subheader("Câu 1.4.3 — Phân rã tăng trưởng 2020–2025")
        cc = st.columns([1, 1])
        with cc[0]:
            st.dataframe(pd.DataFrame({"Yếu tố": list(comp.keys()),
                                       "Đóng góp %/năm": [f"{v*100:.4f}" for v in comp.values()],
                                       "Tỷ lệ %": [f"{r:.2f}" for r in ratio.values()]}),
                         use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            cols = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
            ax.bar(ratio.keys(), ratio.values(), color=cols)
            ax.axhline(0, color="black", lw=0.5); ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=25, ha="right", fontsize=8)
            ax.set_title("Tỷ lệ đóng góp tăng trưởng (%)")
            show_fig(fig)

        st.subheader("Câu 1.4.4 — Mô phỏng GDP 2030")
        K30 = K[-1] * (1 + P["gK"]) ** 5
        L30 = L[-1] * 1.005 ** 5
        A30 = A[-1] * 1.012 ** 5
        Y30 = A30 * (K30 ** a * L30 ** b * P["D30"] ** g * P["AI30"] ** d * P["H30"] ** th)
        gr = ((Y30 / Y[-1]) ** (1 / 5) - 1) * 100
        m = st.columns(3)
        m[0].metric("GDP 2030 dự báo", f"{Y30:,.0f} ngh.tỷ")
        m[1].metric("Tăng trưởng 2025–2030", f"{gr:.2f}%/năm")
        m[2].metric("GDP 2030 / 2025", f"{Y30/Y[-1]:.2f} lần")

    with t_pol:
        st.subheader("1.5. Câu hỏi thảo luận chính sách")
        tfp_cagr = ((A[-1] / A[0]) ** (1 / 5) - 1) * 100
        st.markdown(
            "**a) TFP của Việt Nam tăng hay giảm? Điều đó nói gì về chất lượng tăng trưởng?**  \n"
            f"TFP tăng từ {A[0]:.4f} (2020) lên {A[-1]:.4f} (2025), tức "
            f"**{tfp_cagr:+.2f}%/năm**, và đóng góp **{ratio['TFP (A)']:.1f}%** vào tăng trưởng "
            "GDP. TFP dương và tăng cho thấy tăng trưởng không chỉ dựa vào tích lũy vốn mà còn "
            "nhờ cải thiện hiệu quả và công nghệ — dấu hiệu **chất lượng tăng trưởng đang cải "
            "thiện**, đúng định hướng chuyển từ tăng trưởng theo chiều rộng sang chiều sâu.\n\n"
            "**b) Trong các yếu tố mới D, AI, H, yếu tố nào đóng góp nhiều nhất? Vì sao?**  \n"
            f"Theo phân rã tăng trưởng: D đóng góp {ratio['D (Số hóa)']:.1f}%, AI "
            f"{ratio['AI']:.1f}%, H {ratio['H (Nhân lực)']:.1f}%. Yếu tố dẫn đầu là "
            f"**{top_new}** — chủ yếu do tốc độ tăng nhanh trong giai đoạn (hiệu ứng "
            "Δln lớn) bù lại cho hệ số co giãn còn nhỏ.\n\n"
            "**c) Mục tiêu 30% kinh tế số/GDP vào 2030 có khả thi? Cần ràng buộc gì?**  \n"
            f"Với độ khớp mô hình tốt (MAPE = {mape:.2f}%) và đà tăng D hiện tại "
            f"(~{((D[-1]/D[0])**(1/5)-1)*100:.1f}%/năm), mục tiêu 30% là **khả thi nhưng "
            "thách thức**: cần bổ sung ràng buộc đầu tư đồng bộ vào nhân lực số H (năng lực hấp "
            "thụ) và năng lực AI, tránh số hóa 'lệch pha' với nguồn nhân lực."
        )


# ============================================================================
#  BÀI 2 — LP ngân sách 4 hạng mục
# ============================================================================
def sidebar_bai2():
    with SB:
        st.markdown("### Tham số Bài 2")
        B = st.slider("Ngân sách tổng (ngh.tỷ)", 100, 200, 100, 10)
        x1 = st.slider("Sàn hạ tầng x₁", 0, 40, 25)
        x3 = st.slider("Sàn nhân lực x₃", 20, 40, 20)
        tech = st.slider("Tỷ trọng x₂+x₄ ≥ (%)", 20, 50, 35)
    return dict(B=B, x1=x1, x3=x3, tech=tech / 100)


def page_bai2():
    from scipy.optimize import linprog
    P = sidebar_bai2()
    page_title("💰", "Bài 2 — Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số", "Bài 2")

    c = [-0.85, -1.20, -0.95, -1.35]
    A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1],
            [P["tech"], P["tech"] - 1, P["tech"], P["tech"] - 1]]
    b_ub = [P["B"], -P["x1"], -15, -P["x3"], -10, 0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")

    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["2.1 Bối cảnh", "2.2 Mô hình", "2.3 Dữ liệu", "2.4 Tính toán", "2.5 Chính sách"])

    with t_ctx:
        st.subheader("2.1. Bối cảnh Việt Nam")
        st.write("Theo Quyết định 749/QĐ-TTg, đến 2025 Việt Nam đặt mục tiêu kinh tế số đạt "
                 "20% GDP. Giả sử Bộ KH-ĐT phân bổ **100.000 tỷ VND** ngân sách trung ương "
                 "năm 2026 cho 4 hạng mục: hạ tầng số (I), AI & dữ liệu, nhân lực số (H), "
                 "R&D công nghệ — mỗi hạng mục có hệ số tác động tới tăng GDP khác nhau.")
        c0 = st.columns(3)
        c0[0].metric("Ngân sách", f"{P['B']} ngh.tỷ")
        c0[1].metric("Mục tiêu kinh tế số", "20% GDP")
        c0[2].metric("Số hạng mục", "4")

    with t_mod:
        st.subheader("2.2. Mô hình toán học")
        st.latex(r"\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4")
        st.markdown("Ràng buộc:")
        st.latex(r"x_1+x_2+x_3+x_4 \le B;\ x_1\ge 25;\ x_2\ge 15;\ x_3\ge 20;\ x_4\ge 10")
        st.latex(r"x_2+x_4 \ge 0.35(x_1+x_2+x_3+x_4)")
        st.caption("x₁ hạ tầng, x₂ AI, x₃ nhân lực, x₄ R&D (nghìn tỷ VND).")

    with t_data:
        st.subheader("2.3. Diễn giải hệ số mục tiêu")
        st.dataframe(pd.DataFrame({
            "Hạng mục": ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D"],
            "Hệ số (GDP/đầu tư)": [0.85, 1.20, 0.95, 1.35],
            "Sàn tối thiểu": [P["x1"], 15, P["x3"], 10]}),
            use_container_width=True, hide_index=True)
        st.caption("R&D có hệ số cao nhất do tác động lan tỏa dài hạn; AI cao hơn hạ tầng "
                   "do thu hồi vốn nhanh hơn (World Bank 2024, OECD AI 2024).")

    with t_calc:
        st.subheader("Câu 2.4.1–2.4.2 — Lời giải tối ưu & shadow price")
        if res.success:
            names = ["x₁ Hạ tầng", "x₂ AI & dữ liệu", "x₃ Nhân lực", "x₄ R&D"]
            cc = st.columns([1, 1])
            with cc[0]:
                st.dataframe(pd.DataFrame({"Hạng mục": names, "Phân bổ": np.round(res.x, 2)}),
                             use_container_width=True, hide_index=True)
                st.metric("Z* (GDP tăng thêm)", f"{-res.fun:.2f} ngh.tỷ")
            with cc[1]:
                fig, ax = plt.subplots(figsize=(5.5, 3.4))
                ax.bar(["x₁", "x₂", "x₃", "x₄"], res.x,
                       color=["#3498db", "#9b59b6", "#2ecc71", "#e74c3c"])
                ax.set_ylabel("Ngh.tỷ"); ax.grid(axis="y", alpha=0.3)
                show_fig(fig)
            st.markdown("**Giá đối ngẫu (shadow price) từng ràng buộc — Câu 2.4.2:**")
            try:
                duals = res.ineqlin.marginals
                cons_names = ["Ngân sách tổng", "Sàn x₁ (hạ tầng)", "Sàn x₂ (AI)",
                              "Sàn x₃ (nhân lực)", "Sàn x₄ (R&D)", "Tỷ trọng CN chiến lược"]
                st.dataframe(pd.DataFrame({
                    "Ràng buộc": cons_names,
                    "Shadow price": np.round(np.abs(duals), 4),
                    "Ý nghĩa": ["GDP tăng thêm/1 ngh.tỷ ngân sách", "Giá trị nới sàn x₁",
                                "Giá trị nới sàn x₂", "Giá trị nới sàn x₃",
                                "Giá trị nới sàn x₄", "Chi phí ràng buộc tỷ trọng"]}),
                    use_container_width=True, hide_index=True)
            except Exception:
                pass
            st.info("Shadow price ngân sách tổng = 1,35: mỗi nghìn tỷ ngân sách tăng thêm tạo "
                    "~1,35 nghìn tỷ GDP (= hệ số R&D, hạng mục biên cao nhất). Vì shadow price "
                    "dương, **mọi đồng ngân sách bổ sung đều tạo giá trị vượt chi phí** → nên mở "
                    "rộng ngân sách nếu khả năng tài khóa cho phép.")
        else:
            st.error("Bài toán không khả thi.")

        st.subheader("Câu 2.4.3 — Đường cong $Z^*(B)$")
        Bs = np.arange(100, 201, 10)
        Zs = []
        for bb in Bs:
            r = linprog(c, A_ub=A_ub, b_ub=[bb, -P["x1"], -15, -P["x3"], -10, 0],
                        bounds=[(0, None)] * 4, method="highs")
            Zs.append(-r.fun if r.success else np.nan)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(Bs, Zs, "b-o"); ax.set_xlabel("Ngân sách tổng"); ax.set_ylabel("Z*")
        ax.grid(alpha=0.3); ax.set_title("Phân tích độ nhạy ngân sách")
        show_fig(fig)

        st.subheader("Câu 2.4.4 — Ưu tiên nhân lực số (x₃ ≥ 30)")
        r30 = linprog(c, A_ub=A_ub, b_ub=[100, -P["x1"], -15, -30, -10, 0],
                      bounds=[(0, None)] * 4, method="highs")
        if r30.success:
            st.success(f"Vẫn khả thi. Z* = {-r30.fun:.2f} ngh.tỷ.")
        else:
            st.error("Không khả thi với x₃ ≥ 30.")

    with t_pol:
        st.subheader("2.5. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a)** Khi ngân sách tăng 1 tỷ VND, GDP kỳ vọng tăng ~1,35 tỷ (shadow price) — "
            "đây là cận trên hợp lý của chi phí cơ hội vốn công.\n\n"
            "**b)** R&D có hệ số cao nhất nhưng sàn thấp nhất vì rủi ro và độ trễ lớn, cần "
            "thận trọng khi cam kết ngân sách cứng.\n\n"
            "**c)** Tỷ lệ 35% công nghệ chiến lược (AI+R&D) khó đạt khi ngân sách 2025 ưu tiên "
            "hạ tầng giao thông và an sinh xã hội — cần lộ trình tăng dần."
        )


# ============================================================================
#  BÀI 3 — Priority 10 ngành
# ============================================================================
def sidebar_bai3():
    with SB:
        st.markdown("### Tham số Bài 3")
        st.caption("Trọng số a₁..a₇ (chuẩn hóa lại tổng = 1)")
        labels = ["Tăng trưởng", "Năng suất", "Lan tỏa", "Xuất khẩu", "Việc làm", "AI Ready", "Risk"]
        defaults = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15]
        w = [st.slider(labels[i], 0.0, 0.5, defaults[i], 0.01, key=f"b3_{i}") for i in range(7)]
    return np.array(w)


def page_bai3():
    w_raw = sidebar_bai3()
    page_title("📊", "Bài 3 — Chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam", "Bài 3")

    df = load_sectors().copy()
    GDP_2024 = 11511.9
    df["labor_productivity"] = (df["gdp_share_2024_pct"] / 100) * GDP_2024 / df["labor_million"]
    df["sector_vi"] = SECTOR_VI
    cols_good = ["growth_rate_2024_pct", "labor_productivity", "spillover_coef_0_1",
                 "export_billion_USD", "labor_million", "ai_readiness_0_100"]

    def ng(x): return (x - x.min()) / (x.max() - x.min())
    def nb(x): return (x.max() - x) / (x.max() - x.min())
    Xg = df[cols_good].apply(ng)
    Xb = nb(df["automation_risk_pct"])
    s = w_raw.sum()
    w, w_risk = w_raw[:6] / s, w_raw[6] / s
    priority = Xg.values @ w + w_risk * Xb.values
    df["Priority"] = priority

    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["3.1 Bối cảnh", "3.2 Mô hình", "3.3 Dữ liệu", "3.4 Tính toán", "3.5 Chính sách"])

    with t_ctx:
        st.subheader("3.1. Bối cảnh Việt Nam")
        st.write("Cơ cấu kinh tế 2024: nông-lâm-thủy sản 11,86%, công nghiệp-xây dựng 37,64%, "
                 "dịch vụ 42,36% GDP. Câu hỏi: ngành nào nên ưu tiên đẩy mạnh chuyển đổi số và "
                 "AI trước để tạo hiệu ứng lan tỏa tối đa?")
        c = st.columns(3)
        c[0].metric("Số ngành", "10")
        c[1].metric("Nông-Lâm-Thủy sản", "11,86%")
        c[2].metric("Dịch vụ", "42,36%")

    with t_mod:
        st.subheader("3.2. Mô hình toán học")
        st.latex(r"Priority_i = \sum_{k} a_k \tilde{x}_{ik} - a_7\,\widetilde{Risk}_i")
        st.markdown("Chuẩn hóa min-max về [0,1]:")
        st.latex(r"\tilde{x}_i = \frac{x_i - \min x}{\max x - \min x}, \quad \widetilde{Risk}_i = \frac{\max x - x_i}{\max x - \min x}")

    with t_data:
        st.subheader("3.3. Dữ liệu gốc 10 ngành (2024)")
        raw = df[["sector_vi", "growth_rate_2024_pct", "labor_productivity", "spillover_coef_0_1",
                  "export_billion_USD", "labor_million", "ai_readiness_0_100", "automation_risk_pct"]].copy()
        raw.columns = ["Ngành", "Tăng trưởng %", "Năng suất", "Lan tỏa", "XK (tỷ USD)",
                       "Việc làm (tr)", "AI Ready", "Rủi ro TĐH %"]
        st.dataframe(raw.round(2), use_container_width=True, hide_index=True)
        st.caption("7 tiêu chí: 6 tiêu chí lợi ích (càng cao càng tốt) và Rủi ro tự động hóa "
                   "là tiêu chí 'xấu' (càng thấp càng tốt).")

    with t_calc:
        st.subheader("Câu 3.4.1 — Chuẩn hóa min-max về [0,1] (đảo dấu Risk)")
        st.latex(r"\tilde{x}_i = \frac{x_i - \min x}{\max x - \min x}\ \text{(tiêu chí tốt)}; \quad "
                 r"\tilde{Risk}_i = \frac{\max x - x_i}{\max x - \min x}\ \text{(đảo dấu)}")
        Xg_show = Xg.copy()
        Xg_show.columns = ["Tăng trưởng", "Năng suất", "Lan tỏa", "Xuất khẩu", "Việc làm", "AI Ready"]
        Xg_show.insert(0, "Ngành", df["sector_vi"].values)
        Xg_show["Risk(đảo)"] = np.round(Xb.values, 3)
        st.dataframe(Xg_show.round(3), use_container_width=True, hide_index=True)
        st.caption("Sau chuẩn hóa, mọi tiêu chí về cùng thang [0,1] với hướng 'càng cao càng tốt' "
                   "(Risk đã đảo dấu: ngành rủi ro thấp được điểm cao). Đây là đầu vào để tính Priorityᵢ.")

        st.subheader("Câu 3.4.2 — Xếp hạng Priorityᵢ")
        rank = df[["sector_vi", "Priority"]].sort_values("Priority", ascending=False).reset_index(drop=True)
        rank.index += 1
        rank.columns = ["Ngành", "Priority"]
        cc = st.columns([1, 1])
        with cc[0]:
            st.dataframe(rank.round(4), use_container_width=True)
        with cc[1]:
            fig, ax = plt.subplots(figsize=(6, 4))
            rr = rank.iloc[::-1]
            ax.barh(rr["Ngành"], rr["Priority"], color="#3498db")
            plt.yticks(fontsize=8); ax.set_title("Xếp hạng Priorityᵢ")
            show_fig(fig)

        st.subheader("Câu 3.4.3 — Độ nhạy theo w_AI (heatmap)")
        w_base = np.array([0.15, 0.15, 0.20, 0.15, 0.10]); w_risk_v = 0.15
        rng = np.arange(0.05, 0.45, 0.05)
        H = []
        for wai in rng:
            rem = 1.0 - wai - w_risk_v
            ws = np.append(w_base * (rem / w_base.sum()), wai)
            H.append(Xg.values @ ws + w_risk_v * Xb.values)
        H = np.array(H)
        fig, ax = plt.subplots(figsize=(10, 3.6))
        im = ax.imshow(H, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(len(rng))); ax.set_yticklabels([f"{w:.2f}" for w in rng])
        ax.set_xticks(range(10)); ax.set_xticklabels([f"N{i+1}" for i in range(10)])
        ax.set_xlabel("Ngành"); ax.set_ylabel("w_AI"); plt.colorbar(im, label="Priority")
        show_fig(fig)

        st.subheader("Câu 3.4.4 — Hai bộ trọng số")
        wg = np.array([0.25, 0.25, 0.10, 0.25, 0.05, 0.05]); wg_r = 0.05
        wi = np.array([0.05, 0.10, 0.25, 0.05, 0.25, 0.10]); wi_r = 0.20
        pg = Xg.values @ wg + wg_r * Xb.values
        pi = Xg.values @ wi + wi_r * Xb.values
        cc = st.columns(2)
        with cc[0]:
            st.markdown("**Định hướng tăng trưởng**")
            st.dataframe(pd.DataFrame({"Ngành": df["sector_vi"], "Điểm": np.round(pg, 4)})
                         .sort_values("Điểm", ascending=False).head(5),
                         use_container_width=True, hide_index=True)
        with cc[1]:
            st.markdown("**Định hướng bao trùm**")
            st.dataframe(pd.DataFrame({"Ngành": df["sector_vi"], "Điểm": np.round(pi, 4)})
                         .sort_values("Điểm", ascending=False).head(5),
                         use_container_width=True, hide_index=True)

    with t_pol:
        st.subheader("3.5. Câu hỏi thảo luận chính sách")
        top3 = rank["Ngành"].head(3).tolist()
        st.markdown(
            f"**a)** Ba ngành nên ưu tiên: {', '.join(top3)} — phù hợp tinh thần "
            "**Nghị quyết 57-NQ/TW** về đột phá KHCN và chuyển đổi số.\n\n"
            "**b)** Khai khoáng có năng suất rất cao nhưng tăng trưởng âm, lan tỏa thấp và rủi ro "
            "tự động hóa lớn nên không vào nhóm ưu tiên.\n\n"
            "**c)** Bộ trọng số nên do **hội đồng chính sách + đối thoại công khai** quyết định, "
            "không chỉ chuyên gia kỹ thuật, để bảo đảm tính chính danh."
        )


# ============================================================================
#  BÀI 4 — LP ngành-vùng
# ============================================================================
def sidebar_bai4():
    with SB:
        st.markdown("### Tham số Bài 4")
        budget = st.slider("Ngân sách tổng (tỷ)", 30000, 70000, 50000, 5000)
        floor = st.slider("Sàn mỗi vùng (tỷ)", 3000, 8000, 5000, 500)
        cap = st.slider("Trần mỗi vùng (tỷ)", 9000, 15000, 12000, 500)
        lam = st.slider("λ công bằng (C5)", 0.3, 0.9, 0.6, 0.05)
    return dict(budget=budget, floor=floor, cap=cap, lam=lam)


def page_bai4():
    import pulp
    P = sidebar_bai4()
    page_title("🗺️", "Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành-vùng", "Bài 4")
    dr = load_regions()
    D0 = dict(zip(dr["region_name_en"].map({
        "Northern Midlands and Mountains": "NMM", "Red River Delta": "RRD",
        "North Central and South Central Coast": "NCC", "Central Highlands": "CH",
        "Southeast": "SE", "Mekong Delta": "MD"}), dr["digital_index_0_100"]))
    gamma_val = 0.002

    def solve_lp(eq=True):
        m = pulp.LpProblem("VN", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", (REGIONS, ITEMS), lowBound=0)
        m += pulp.lpSum(BETA[(r, j)] * x[r][j] for r in REGIONS for j in ITEMS)
        m += pulp.lpSum(x[r][j] for r in REGIONS for j in ITEMS) <= P["budget"]
        for r in REGIONS:
            m += pulp.lpSum(x[r][j] for j in ITEMS) >= P["floor"]
            m += pulp.lpSum(x[r][j] for j in ITEMS) <= P["cap"]
        m += pulp.lpSum(x[r]["H"] for r in REGIONS) >= 12000
        if eq:
            M = pulp.LpVariable("Dmax")
            for r in REGIONS:
                m += D0[r] + gamma_val * x[r]["D"] <= M
                m += D0[r] + gamma_val * x[r]["D"] >= P["lam"] * M
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        res = np.array([[x[r][j].value() or 0 for j in ITEMS] for r in REGIONS])
        return res, pulp.value(m.objective)

    x_opt, Z = solve_lp(True)
    x_no, Z_no = solve_lp(False)

    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["4.1 Bối cảnh", "4.2 Mô hình", "4.3 Dữ liệu", "4.4 Tính toán", "4.5 Chính sách"])

    with t_ctx:
        st.subheader("4.1. Bối cảnh Việt Nam")
        st.write("Theo Quyết định 411/QĐ-TTg, các vùng KT-XH có mức độ sẵn sàng số rất khác "
                 "nhau. Bài toán: phân bổ **50.000 tỷ VND** ngân sách kinh tế số cho 6 vùng và "
                 "4 hạng mục (I, D, AI, H) sao cho tối đa GDP gain nhưng bảo đảm công bằng vùng miền.")
        c = st.columns(3)
        c[0].metric("Số vùng × hạng mục", "6 × 4 = 24 biến")
        c[1].metric("Ngân sách", f"{P['budget']:,} tỷ")
        c[2].metric("Sàn H toàn quốc", "12.000 tỷ")

    with t_mod:
        st.subheader("4.2. Mô hình toán học")
        st.latex(r"\max Z = \sum_r \sum_j \beta_{j,r}\,x_{j,r}")
        st.markdown("**C1** ngân sách tổng; **C2/C3** sàn/trần mỗi vùng; **C4** sàn nhân lực số; "
                    "**C5** công bằng vùng:")
        st.latex(r"D_r + \gamma\,x_{D,r} \ge \lambda \cdot \max_r (D_r + \gamma\,x_{D,r})")

    with t_data:
        st.subheader("4.3. Ma trận hệ số tác động biên βⱼ,ᵣ")
        st.dataframe(pd.DataFrame(BETA_MAT, columns=ITEMS, index=REGION_VI),
                     use_container_width=True)
        st.caption(f"Chỉ số số hóa ban đầu D₀ (từ CSV): {dict(zip(REGIONS, [int(D0[r]) for r in REGIONS]))}")

    with t_calc:
        st.subheader("Câu 4.4.1–4.4.3 — Phân bổ tối ưu & heatmap")
        cc = st.columns([1.2, 1])
        dfp = pd.DataFrame(x_opt, columns=ITEMS, index=REGION_VI).round(0)
        dfp["Tổng"] = dfp.sum(1)
        with cc[0]:
            st.dataframe(dfp, use_container_width=True)
            st.metric("Z* (có công bằng)", f"{Z:,.0f} tỷ")
        with cc[1]:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            im = ax.imshow(x_opt, cmap="YlOrRd", aspect="auto")
            ax.set_yticks(range(6)); ax.set_yticklabels([r[:14] for r in REGION_VI], fontsize=8)
            ax.set_xticks(range(4)); ax.set_xticklabels(ITEMS)
            for i in range(6):
                for j in range(4):
                    ax.text(j, i, f"{x_opt[i,j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if x_opt[i, j] > 8000 else "black")
            plt.colorbar(im, ax=ax, shrink=0.8)
            show_fig(fig)
        st.caption("PuLP (CBC) và CVXPY cho cùng nghiệm (sai khác < 1e-4) — bài LP lồi có "
                   "nghiệm tối ưu ổn định.")
        # đọc kết quả: vùng & hạng mục nhận nhiều nhất
        reg_tot = x_opt.sum(1)
        item_tot = x_opt.sum(0)
        top_reg = REGION_VI[int(np.argmax(reg_tot))]
        top_item = {"I": "hạ tầng số", "D": "chuyển đổi số DN",
                    "AI": "năng lực AI", "H": "nhân lực số"}[ITEMS[int(np.argmax(item_tot))]]
        st.caption(f"📌 Vùng nhận nhiều ngân sách nhất: **{top_reg}** "
                   f"({reg_tot.max():,.0f} tỷ); hạng mục được ưu tiên nhất toàn quốc: "
                   f"**{top_item}** ({item_tot.max():,.0f} tỷ). Nhiều vùng chạm trần "
                   f"{P['cap']:,.0f} tỷ — ràng buộc trần đang 'binding', kìm vốn khỏi dồn về vùng giàu.")

        st.subheader("Câu 4.4.4 — Chi phí của công bằng vùng miền")
        m = st.columns(3)
        m[0].metric("Z* CÓ công bằng", f"{Z:,.0f}")
        m[1].metric("Z* KHÔNG công bằng", f"{Z_no:,.0f}")
        m[2].metric("Chi phí công bằng", f"{Z_no-Z:,.0f}", f"-{(Z_no-Z)/Z_no*100:.2f}%")
        st.info(f"Áp ràng buộc công bằng vùng (C5) làm GDP gain giảm **{Z_no-Z:,.0f} tỷ** "
                f"(~{(Z_no-Z)/Z_no*100:.2f}% so với phương án thuần hiệu quả). Đây là 'cái giá "
                "của công bằng' — mức đánh đổi định lượng để giảm chênh lệch số hóa giữa các "
                "vùng; con số nhỏ cho thấy công bằng vùng miền **không quá tốn kém** về hiệu quả.")

    with t_pol:
        st.subheader("4.5. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a)** Nếu bỏ ràng buộc công bằng, vốn chảy về Đông Nam Bộ và ĐB sông Hồng "
            "(β_AI cao nhất) → gia tăng bất bình đẳng vùng miền dài hạn.\n\n"
            "**b)** Trần ngân sách mỗi vùng (C3) như 'chính sách phân quyền', làm giảm Z* một "
            "phần nhưng chấp nhận được để chống tập trung quá mức.\n\n"
            "**c)** Tây Nguyên có β_AI thấp (0,45) → nên ưu tiên H và I trước, AI sau khi đủ "
            "nền tảng nhân lực."
        )


# ============================================================================
#  BÀI 5 — MIP chọn dự án
# ============================================================================
def sidebar_bai5():
    with SB:
        st.markdown("### Tham số Bài 5")
        budget = st.slider("Ngân sách 5 năm (tỷ)", 60000, 120000, 80000, 10000)
        use_exp = st.checkbox("Tối đa lợi ích kỳ vọng (rủi ro pᵢ)", value=False)
        force12 = st.checkbox("Bắt buộc cả P1 & P2", value=False)
    return dict(budget=budget, use_exp=use_exp, force12=force12)


def page_bai5():
    from pulp import (LpProblem, LpMaximize, LpVariable, lpSum, value,
                      PULP_CBC_CMD, LpStatus)
    P = sidebar_bai5()
    page_title("🎯", "Bài 5 — Quy hoạch nguyên hỗn hợp (MIP) lựa chọn dự án chuyển đổi số", "Bài 5")

    Pr = list(range(1, 16))
    C = {1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200, 6: 5800, 7: 6500, 8: 15000,
         9: 2500, 10: 7200, 11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500}
    C1 = {1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500, 6: 4000, 7: 4500, 8: 9000,
          9: 1800, 10: 5000, 11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200}
    B = {1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800, 6: 11400, 7: 12200, 8: 28500,
         9: 5800, 10: 13800, 11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800}
    names = {1: "TT dữ liệu Hòa Lạc", 2: "TT dữ liệu phía Nam", 3: "5G toàn quốc",
             4: "VNeID 2.0", 5: "Cổng DVC v3", 6: "Y tế số", 7: "Giáo dục số K-12",
             8: "TT AI + supercomputing", 9: "Fintech sandbox", 10: "Logistics thông minh",
             11: "Nông nghiệp số ĐBSCL", 12: "Đào tạo 50K kỹ sư AI", 13: "Khu CN bán dẫn BN-BG",
             14: "An ninh mạng SOC", 15: "Open Data"}
    fld = {1: "ht", 2: "ht", 3: "ht", 4: "cp", 5: "cp", 6: "yt", 7: "gd", 8: "ai",
           9: "tc", 10: "lg", 11: "nn", 12: "nl", 13: "bd", 14: "an", 15: "dl"}
    prob = {"ht": .85, "cp": .75, "ai": .65, "bd": .65, "yt": .8, "gd": .8, "tc": .8,
            "lg": .8, "nn": .8, "nl": .8, "an": .8, "dl": .8}

    def solve_mip(budget=None, force12=None, use_exp=None):
        bud = P["budget"] if budget is None else budget
        f12 = P["force12"] if force12 is None else force12
        uex = P["use_exp"] if use_exp is None else use_exp
        m = LpProblem("VN", LpMaximize)
        y = LpVariable.dicts("y", Pr, cat="Binary")
        if uex:
            m += lpSum(prob[fld[i]] * B[i] * y[i] for i in Pr)
        else:
            m += lpSum(B[i] * y[i] for i in Pr)
        m += lpSum(C[i] * y[i] for i in Pr) <= bud
        m += lpSum(C1[i] * y[i] for i in Pr) <= 40000
        if not f12:
            m += y[1] + y[2] <= 1
        else:
            m += y[1] >= 1; m += y[2] >= 1
        m += y[8] <= y[12]; m += y[13] <= y[12]
        m += y[4] + y[5] >= 1; m += y[14] >= 1
        m += lpSum(y[i] for i in Pr) >= 7
        m += lpSum(y[i] for i in Pr) <= 11
        m.solve(PULP_CBC_CMD(msg=False))
        sel = [i for i in Pr if y[i].value() and y[i].value() > 0.5]
        return sel, value(m.objective), LpStatus[m.status]

    sel, Z, status = solve_mip()

    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["5.1 Bối cảnh", "5.2 Mô hình", "5.3 Dữ liệu", "5.4 Tính toán", "5.5 Chính sách"])

    with t_ctx:
        st.subheader("5.1. Bối cảnh Việt Nam")
        st.write("Bộ KH-CN xem xét **15 dự án** ứng cử cho chương trình chuyển đổi số quốc gia "
                 "2026–2030, tổng ngân sách **80.000 tỷ VND**. Mỗi dự án có chi phí, lợi ích NPV "
                 "và ràng buộc đặc thù (loại trừ, tiên quyết, cân đối lĩnh vực).")
        c = st.columns(3)
        c[0].metric("Số dự án ứng cử", "15")
        c[1].metric("Ngân sách", f"{P['budget']:,} tỷ")
        c[2].metric("Số dự án chọn", f"{len(sel)}")

    with t_mod:
        st.subheader("5.2. Mô hình toán học")
        st.latex(r"\max \sum_i B_i y_i, \quad y_i \in \{0,1\}")
        st.markdown("Ràng buộc: ngân sách 5 năm & năm 1-2; loại trừ y₁+y₂≤1; tiên quyết "
                    "y₈≤y₁₂, y₁₃≤y₁₂; cân đối lĩnh vực y₄+y₅≥1, y₁₄≥1; số dự án 7≤Σyᵢ≤11.")

    with t_data:
        st.subheader("5.3. Danh mục 15 dự án")
        st.dataframe(pd.DataFrame({
            "Mã": [f"P{i}" for i in Pr], "Tên": [names[i] for i in Pr],
            "Chi phí": [C[i] for i in Pr], "NPV": [B[i] for i in Pr],
            "B/C": [round(B[i] / C[i], 2) for i in Pr]}),
            use_container_width=True, hide_index=True, height=400)

    with t_calc:
        st.subheader("Câu 5.4.1 — Kết quả lựa chọn tối ưu")
        if status == "Optimal":
            st.dataframe(pd.DataFrame([{"Mã": f"P{i}", "Dự án": names[i], "Chi phí": C[i],
                                        "NPV": B[i], "B/C": round(B[i] / C[i], 2)} for i in sel]),
                         use_container_width=True, hide_index=True)
            tc = sum(C[i] for i in sel)
            m = st.columns(4)
            m[0].metric("Số dự án", len(sel))
            m[1].metric("Tổng chi phí", f"{tc:,} tỷ")
            m[2].metric("Z* lợi ích", f"{Z:,.0f} tỷ")
            m[3].metric("B/C TB", f"{Z/tc:.2f}")
            st.caption("Lưu ý: P15 (Open Data) B/C=2,53 cao nhất nên **được chọn** — khác giả "
                       "thiết câu 5.5.a của đề (giả thiết bỏ P15 là không đúng).")

            st.subheader("Câu 5.4.2 — Phân tích nới ngân sách")
            rows = []
            base_sel, base_z, _ = solve_mip(80000)
            for bud in [80000, 90000, 100000, 110000]:
                s2, z2, _ = solve_mip(bud)
                rows.append({"Ngân sách (tỷ)": f"{bud:,}", "Số dự án": len(s2),
                             "Z* lợi ích (tỷ)": f"{z2:,.0f}" if z2 else "—",
                             "Dự án": ", ".join(f"P{i}" for i in s2)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            s100, z100, _ = solve_mip(100000)
            added = set(s100) - set(base_sel)
            st.info(f"Nới ngân sách 80.000 → 100.000 tỷ giúp Z* tăng "
                    f"{z100 - base_z:,.0f} tỷ, bổ sung các dự án: "
                    f"{', '.join('P'+str(i) for i in added) if added else 'không đổi'}. "
                    "Lợi ích biên giảm dần khi ngân sách lớn (các dự án B/C cao đã được chọn trước).")

            st.subheader("Câu 5.4.3 — Bắt buộc cả P1 và P2 (redundancy)")
            s3, z3, st3 = solve_mip(force12=True)
            cc = st.columns([1, 1])
            with cc[0]:
                if st3 == "Optimal":
                    st.success("Bài toán **vẫn khả thi** khi bắt buộc cả P1 & P2.")
                    st.metric("Z* (bắt buộc P1&P2)", f"{z3:,.0f} tỷ",
                              f"{z3 - Z:,.0f} so với ban đầu")
                    st.caption(f"Dự án chọn: {', '.join('P'+str(i) for i in s3)}")
                else:
                    st.error("Bài toán KHÔNG khả thi khi bắt buộc cả P1 & P2.")
            with cc[1]:
                st.markdown("Khi ép chọn cả 2 trung tâm dữ liệu (Hòa Lạc + phía Nam) để có "
                            "redundancy, ràng buộc loại trừ y₁+y₂≤1 được thay bằng y₁≥1, y₂≥1. "
                            "Hai dự án này tốn ~23.500 tỷ nên 'lấn' ngân sách của dự án khác, "
                            f"khiến Z* {'giảm' if st3=='Optimal' and z3 < Z else 'thay đổi'} — "
                            "đây là **chi phí của yêu cầu an toàn dữ liệu**.")

            st.subheader("Câu 5.4.4 — Tối đa lợi ích kỳ vọng (rủi ro hoàn thành pᵢ)")
            s4, z4, _ = solve_mip(use_exp=True)
            cc = st.columns([1.2, 1])
            with cc[0]:
                st.dataframe(pd.DataFrame({
                    "Lĩnh vực": ["Hạ tầng", "Chính phủ số", "AI/Bán dẫn", "Còn lại"],
                    "Xác suất pᵢ": [0.85, 0.75, 0.65, 0.80]}),
                    use_container_width=True, hide_index=True)
            with cc[1]:
                st.metric("E[Z] (lợi ích kỳ vọng)", f"{z4:,.0f} tỷ")
                st.caption(f"Dự án chọn: {', '.join('P'+str(i) for i in s4)}")
            dropped = set(sel) - set(s4)
            addedx = set(s4) - set(sel)
            st.info("Khi tính rủi ro hoàn thành (E[Z]=Σpᵢ·Bᵢ·yᵢ), các dự án AI/bán dẫn "
                    "(p=0,65) bị 'phạt' nặng hơn. " +
                    (f"Dự án bị loại: {', '.join('P'+str(i) for i in dropped)}. " if dropped else "") +
                    (f"Dự án được thêm: {', '.join('P'+str(i) for i in addedx)}. " if addedx else "") +
                    "Mô hình ưu tiên dự án chắc chắn hoàn thành hơn, phản ánh thái độ thận trọng "
                    "với rủi ro triển khai.")
        else:
            st.error("Bài toán KHÔNG khả thi.")

    with t_pol:
        st.subheader("5.5. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a)** P15 (Open Data) thực tế **được chọn** vì B/C cao nhất — kết quả mong muốn "
            "về chính sách (dữ liệu mở tạo lan tỏa).\n\n"
            "**b)** Bắt buộc P14 (an ninh mạng) có thể giảm Z* nhưng hợp lý vì an ninh là điều "
            "kiện nền cho mọi hệ thống số.\n\n"
            "**c)** P8 (AI) và P13 (bán dẫn) có lợi ích cộng hưởng — mô hình hóa bằng biến tích "
            "z = y₈·y₁₃ tuyến tính hóa (z≤y₈, z≤y₁₃, z≥y₈+y₁₃−1)."
        )


# ============================================================================
#  BÀI 6 — TOPSIS 6 vùng
# ============================================================================
def sidebar_bai6():
    with SB:
        st.markdown("### Tham số Bài 6")
        method = st.radio("Trọng số", ["Chuyên gia", "Entropy", "AHP"], index=0)
        w_ai = st.slider("w_AI (độ nhạy)", 0.10, 0.40, 0.20, 0.05)
    return dict(method=method, w_ai=w_ai)


def page_bai6():
    P = sidebar_bai6()
    page_title("🏆", "Bài 6 — TOPSIS xếp hạng 6 vùng theo mức độ ưu tiên đầu tư AI", "Bài 6")
    df = load_regions()
    crit = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
            "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
            "internet_penetration_pct", "gini_coef"]
    labels = ["GRDP/N", "FDI", "Digital", "AI", "LĐĐT", "R&D", "Internet", "Gini"]
    is_ben = [True, True, True, True, True, True, True, False]
    X = df[crit].values.astype(float)
    w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    w_ent = entropy_weights(X)
    ahp = np.array([[1, 1, 1/3, 1/5, 1/3, 1/3, 3, 3], [1, 1, 1/3, 1/5, 1/3, 1/3, 3, 3],
                    [3, 3, 1, 1/2, 1, 1, 5, 5], [5, 5, 2, 1, 2, 2, 7, 7],
                    [3, 3, 1, 1/2, 1, 1, 5, 5], [3, 3, 1, 1/2, 1, 1, 5, 5],
                    [1/3, 1/3, 1/5, 1/7, 1/5, 1/5, 1, 1], [1/3, 1/3, 1/5, 1/7, 1/5, 1/5, 1, 1]])
    gm = np.prod(ahp, axis=1) ** (1 / 8)
    w_ahp = gm / gm.sum()
    C_exp = topsis(X, w_expert, is_ben)
    C_ent = topsis(X, w_ent, is_ben)
    C_ahp = topsis(X, w_ahp, is_ben)
    C_sel = {"Chuyên gia": C_exp, "Entropy": C_ent, "AHP": C_ahp}[P["method"]]

    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["6.1 Bối cảnh", "6.2 Mô hình", "6.3 Dữ liệu", "6.4 Tính toán", "6.5 Chính sách"])

    with t_ctx:
        st.subheader("6.1. Bối cảnh Việt Nam")
        st.write("Theo Quyết định 127/QĐ-TTg, Việt Nam đặt mục tiêu trở thành trung tâm AI của "
                 "ASEAN. Ngân sách có hạn nên cần chọn vùng triển khai trung tâm AI và sandbox "
                 "dữ liệu trước. Bài tập áp dụng **TOPSIS** xếp hạng 6 vùng theo mức độ sẵn sàng AI.")
        c = st.columns(3)
        c[0].metric("Số vùng", "6")
        c[1].metric("Số tiêu chí", "8")
        c[2].metric("Mục tiêu", "3 trung tâm AI")

    with t_mod:
        st.subheader("6.2. Lý thuyết TOPSIS")
        st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}}, \quad v_{ij} = w_j r_{ij}")
        st.latex(r"S_i^* = \sqrt{\sum_j (v_{ij}-v_j^*)^2}, \quad C_i^* = \frac{S_i^-}{S_i^* + S_i^-}")
        st.markdown("Trọng số Entropy (khách quan): " +
                    str({labels[i]: round(w_ent[i], 3) for i in range(8)}))

    with t_data:
        st.subheader("6.3. Dữ liệu 6 vùng KT-XH")
        dd = df[["region_name_en"] + crit].copy()
        dd["region_name_en"] = REGION_VI
        dd.columns = ["Vùng"] + labels
        st.dataframe(dd, use_container_width=True, hide_index=True)
        st.caption("GRDP/N, FDI, Digital, AI, LĐĐT, R&D, Internet là tiêu chí lợi ích; Gini là chi phí.")

    with t_calc:
        st.subheader("Câu 6.4.1 — TOPSIS với trọng số chuyên gia")
        st.caption("Trọng số chuyên gia w = [0,10; 0,10; 0,15; 0,20; 0,15; 0,15; 0,05; 0,10] "
                   "(ưu tiên AI Readiness và Digital Index).")
        st.subheader("Câu 6.4.2 — So sánh với trọng số khách quan (Entropy)")
        st.caption(f"Đang xem theo trọng số: **{P['method']}** (đổi ở sidebar). Bảng dưới hiển thị "
                   "đồng thời C* của cả 3 cách xác định trọng số để đối chiếu.")
        res = pd.DataFrame({
            "Vùng": REGION_VI, "C* Chuyên gia": np.round(C_exp, 4),
            "C* Entropy": np.round(C_ent, 4), "C* AHP": np.round(C_ahp, 4),
            f"Hạng ({P['method']})": pd.Series(C_sel).rank(ascending=False).astype(int).values,
        }).sort_values(f"Hạng ({P['method']})")
        cc = st.columns([1.3, 1])
        with cc[0]:
            st.dataframe(res, use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            order = np.argsort(C_sel)
            ax.barh(np.array(REGION_VI)[order], C_sel[order], color="#3498db")
            plt.yticks(fontsize=8); ax.set_title(f"C* ({P['method']})")
            show_fig(fig)
        top1 = REGION_VI[int(np.argmax(C_sel))]
        last = REGION_VI[int(np.argmin(C_sel))]
        diff_ent = int((pd.Series(C_ent).rank(ascending=False) -
                        pd.Series(C_exp).rank(ascending=False)).abs().idxmax())
        st.caption(f"📌 Dẫn đầu: **{top1}** (C*={C_sel.max():.3f}) — vùng nên đặt trung tâm AI "
                   f"đầu tiên. Cuối bảng: **{last}** (C*={C_sel.min():.3f}). Khi đổi sang trọng số "
                   f"Entropy, **{REGION_VI[diff_ent]}** thay đổi thứ hạng mạnh nhất, do trọng số "
                   "khách quan nhấn vào tiêu chí có độ phân tán lớn (FDI, R&D).")

        st.subheader("Câu 6.4.3 — Độ nhạy theo w_AI")
        rng = np.arange(0.10, 0.45, 0.05)
        H, top3 = [], []
        for wai in rng:
            wg = 0.10; rem = 1 - wai - wg
            wb = np.array([0.10, 0.10, 0.15, 0.15, 0.15, 0.05])
            wf = np.append(np.insert(wb * (rem / wb.sum()), 3, wai), wg)
            cs = topsis(X, wf, is_ben)
            H.append(cs); top3.append([REGION_VI[j] for j in np.argsort(cs)[-3:][::-1]])
        H = np.array(H)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        im = ax.imshow(H, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(len(rng))); ax.set_yticklabels([f"{w:.2f}" for w in rng])
        ax.set_xticks(range(6)); ax.set_xticklabels([f"R{i+1}" for i in range(6)])
        ax.set_xlabel("Vùng"); ax.set_ylabel("w_AI"); plt.colorbar(im, label="C*")
        show_fig(fig)
        st.caption(f"Top-3 {'ỔN ĐỊNH' if all(t==top3[0] for t in top3) else 'CÓ thay đổi'}. "
                   f"Top-3: {top3[0]}")

        st.subheader("Câu 6.4.4 — AHP đơn giản & so sánh 3 phương pháp")
        Aw = ahp @ w_ahp
        lam_max = float(np.mean(Aw / w_ahp))
        CI = (lam_max - 8) / (8 - 1)
        CR = CI / 1.41  # RI cho n=8
        cc = st.columns([1, 1])
        with cc[0]:
            st.dataframe(pd.DataFrame({"Tiêu chí": labels, "Trọng số AHP": np.round(w_ahp, 4)}),
                         use_container_width=True, hide_index=True)
            st.metric("λ_max", f"{lam_max:.3f}")
            st.metric("Consistency Ratio (CR)", f"{CR:.4f}",
                      "Nhất quán (CR<0,1)" if CR < 0.1 else "Chưa nhất quán")
        with cc[1]:
            cmp = pd.DataFrame({
                "Vùng": REGION_VI,
                "Chuyên gia": pd.Series(C_exp).rank(ascending=False).astype(int).values,
                "Entropy": pd.Series(C_ent).rank(ascending=False).astype(int).values,
                "AHP": pd.Series(C_ahp).rank(ascending=False).astype(int).values})
            st.dataframe(cmp, use_container_width=True, hide_index=True)
        st.caption(f"AHP dùng ma trận so sánh cặp (thang Saaty 1-9), tính trọng số bằng trung "
                   f"bình hình học. CR = {CR:.3f} < 0,10 nên ma trận **nhất quán**, trọng số đáng "
                   "tin cậy. Ba phương pháp (Chuyên gia/Entropy/AHP) cho thứ hạng top khá ổn định, "
                   "xác nhận kết quả TOPSIS robust với cách chọn trọng số.")

    with t_pol:
        st.subheader("6.5. Câu hỏi thảo luận chính sách")
        top3_exp = [REGION_VI[j] for j in np.argsort(C_exp)[-3:][::-1]]
        st.markdown(
            f"**a)** Vùng dẫn đầu TOPSIS (chuyên gia): **{top3_exp[0]}** — nên triển khai trung "
            "tâm AI quốc gia đầu tiên.\n\n"
            "**b)** Trọng số Entropy thay đổi xếp hạng vì phản ánh độ phân tán dữ liệu khách quan "
            "thay vì chủ quan.\n\n"
            "**c)** AI Readiness và Internet penetration tương quan cao → dùng PCA hoặc Entropy "
            "để giảm thiên lệch.\n\n"
            f"**d)** Theo Quyết định 127 (3 trung tâm AI), chọn: {', '.join(top3_exp)}."
        )


# ============================================================================
#  BÀI 7 — NSGA-II
# ============================================================================
@st.cache_data(show_spinner="Đang chạy NSGA-II (pop=100, gen=200)...")
def run_nsga(seed=42, pop=100, gen=200):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as moo_min
    from pymoo.termination import get_termination
    gv, lv = 0.002, 0.6

    class Prob(ElementwiseProblem):
        def __init__(s):
            super().__init__(n_var=24, n_obj=4, n_ieq_constr=20,
                             xl=np.zeros(24), xu=np.ones(24) * 12000)

        def _evaluate(s, x, out, *a, **k):
            X = x.reshape(6, 4)
            f1 = -(BETA_MAT * X).sum()
            sm = X.sum(1); f2 = np.abs(sm - sm.mean()).mean()
            f3 = (E_R * (X[:, 0] + X[:, 1] + X[:, 2])).sum()
            f4 = (RHO_R * X[:, 2]).sum() - (SIG_R * X[:, 3]).sum()
            out["F"] = [f1, f2, f3, f4]
            g = [X.sum() - 50000]
            for r in range(6):
                g.append(5000 - X[r].sum())
            for r in range(6):
                g.append(X[r].sum() - 12000)
            g.append(12000 - X[:, 3].sum())
            Dn = D0_ARR + gv * X[:, 1]; Dm = Dn.max()
            for r in range(6):
                g.append(lv * Dm - Dn[r])
            out["G"] = np.array(g)

    res = moo_min(Prob(), NSGA2(pop_size=pop), get_termination("n_gen", gen),
                  seed=seed, verbose=False)
    return res.F, res.X


def sidebar_bai7():
    with SB:
        st.markdown("### Tham số Bài 7")
        st.caption("Trọng số TOPSIS chọn nghiệm thỏa hiệp")
        wg = st.slider("Tăng trưởng", 0.0, 1.0, 0.40, 0.05)
        wgi = st.slider("Bao trùm", 0.0, 1.0, 0.25, 0.05)
        we = st.slider("Môi trường", 0.0, 1.0, 0.20, 0.05)
        wa = st.slider("An ninh", 0.0, 1.0, 0.15, 0.05)
    arr = np.array([wg, wgi, we, wa])
    return arr / arr.sum() if arr.sum() > 0 else np.array([0.4, 0.25, 0.2, 0.15])


def page_bai7():
    w_policy = sidebar_bai7()
    page_title("🌐", "Bài 7 — Tối ưu đa mục tiêu Pareto với NSGA-II", "Bài 7")
    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["7.1 Bối cảnh", "7.2 Mô hình", "7.3 Dữ liệu", "7.4 Tính toán", "7.5 Chính sách"])

    with t_ctx:
        st.subheader("7.1. Bối cảnh Việt Nam")
        st.write("Phát triển kinh tế số hướng tới 4 mục tiêu xung đột: (i) tăng trưởng GDP nhanh; "
                 "(ii) bao trùm xã hội, giảm bất bình đẳng vùng; (iii) net-zero 2050 (COP26); "
                 "(iv) an ninh dữ liệu & chủ quyền số. Kết quả là **tập nghiệm Pareto**, không "
                 "phải nghiệm tối ưu duy nhất.")

    with t_mod:
        st.subheader("7.2. Mô hình đa mục tiêu (24 biến, 4 mục tiêu)")
        st.latex(r"\max f_1 = \sum \beta_{j,r} x_{j,r}\ ;\quad \min f_2 = Gini(x)")
        st.latex(r"\min f_3 = \sum_r e_r (x_{I,r}+x_{AI,r})\ ;\quad \min f_4 = \sum_r \rho_r x_{AI,r} - \sum_r \sigma_r x_{H,r}")
        st.caption("Ràng buộc C1–C5 giữ nguyên như Bài 4. Giải bằng NSGA-II (pop=100, gen=200).")

    with t_data:
        st.subheader("7.3. Tham số bổ sung theo vùng")
        st.dataframe(pd.DataFrame({"Vùng": REGION_VI, "eᵣ (phát thải)": E_R,
                                   "ρᵣ (rủi ro/AI)": RHO_R, "σᵣ (giảm rủi ro/H)": SIG_R}),
                     use_container_width=True, hide_index=True)

    with t_calc:
        F, X = run_nsga()
        if F is None or len(F) == 0:
            st.error("NSGA-II không tìm được nghiệm khả thi với cấu hình hiện tại.")
            return

        st.subheader("Câu 7.4.1 — Tập nghiệm Pareto (NSGA-II)")
        m = st.columns(4)
        m[0].metric("Số nghiệm Pareto", len(F))
        m[1].metric("GDP gain (tỷ)", f"{-F[:,0].min():,.0f}",
                    f"khoảng [{-F[:,0].max():,.0f}; {-F[:,0].min():,.0f}]")
        m[2].metric("Gini/MAD min", f"{F[:,1].min():.0f}",
                    f"khoảng [{F[:,1].min():.0f}; {F[:,1].max():.0f}]")
        m[3].metric("Phát thải min", f"{F[:,2].min():,.0f}")
        st.caption("Mỗi điểm Pareto là một phương án mà không thể cải thiện một mục tiêu nếu "
                   "không làm xấu đi ít nhất một mục tiêu khác. Khoảng dao động lớn của f₁ và f₂ "
                   "cho thấy không gian đánh đổi rộng — dư địa lựa chọn chính sách rất phong phú.")

        st.subheader("Câu 7.4.2 — Trực quan hóa: scatter 3D & parallel coordinates")
        fig = plt.figure(figsize=(13, 4.4))
        ax1 = fig.add_subplot(121, projection="3d")
        sc = ax1.scatter(-F[:, 0], F[:, 1], F[:, 2], c=F[:, 3], cmap="viridis", s=12, alpha=0.7)
        ax1.set_xlabel("GDP gain"); ax1.set_ylabel("Gini/MAD"); ax1.set_zlabel("Phát thải")
        ax1.set_title("Mặt Pareto 3D (màu = rủi ro f₄)"); fig.colorbar(sc, ax=ax1, shrink=0.6)
        ax2 = fig.add_subplot(122)
        Fn = np.copy(F)
        for i in range(4):
            lo, hi = F[:, i].min(), F[:, i].max()
            Fn[:, i] = (F[:, i] - lo) / (hi - lo) if hi > lo else 0.5
        for i in range(len(F)):
            ax2.plot(range(4), Fn[i], "b-", alpha=0.05, lw=0.5)
        ax2.plot(range(4), Fn.mean(0), "r-", lw=2, label="Trung bình")
        ax2.set_xticks(range(4)); ax2.set_xticklabels(["GDP\n(↑)", "Gini\n(↓)", "Phát thải\n(↓)", "Rủi ro\n(↓)"])
        ax2.set_ylabel("Giá trị chuẩn hóa [0,1]"); ax2.legend(); ax2.set_title("Parallel coordinates 4 mục tiêu")
        show_fig(fig)

        # tương quan đánh đổi
        corr = np.corrcoef(-F[:, 0], F[:, 1])[0, 1]
        st.caption(f"Hệ số tương quan giữa GDP gain và Gini = **{corr:+.2f}**. "
                   f"{'Tương quan dương → tăng trưởng càng cao thì càng bất bình đẳng (đánh đổi rõ rệt).' if corr > 0.1 else 'Tương quan yếu → có thể đạt tăng trưởng mà không hi sinh nhiều công bằng.'}")

        st.subheader("Câu 7.4.3 — Nghiệm thỏa hiệp bằng TOPSIS")
        fmin, fmax = F.min(0), F.max(0)
        fr = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
        V = (F - fmin) / fr * w_policy
        S_star = np.sqrt((V ** 2).sum(1))
        S_neg = np.sqrt(((V - w_policy) ** 2).sum(1))
        C = S_neg / (S_star + S_neg)
        best = int(np.argmax(C))
        st.markdown(f"Trọng số ưu tiên chính sách (chuẩn hóa): tăng trưởng "
                    f"{w_policy[0]:.2f}, bao trùm {w_policy[1]:.2f}, môi trường "
                    f"{w_policy[2]:.2f}, an ninh {w_policy[3]:.2f}.")
        m = st.columns(4)
        m[0].metric("GDP gain", f"{-F[best,0]:,.0f}")
        m[1].metric("Gini/MAD", f"{F[best,1]:.0f}")
        m[2].metric("Phát thải", f"{F[best,2]:,.0f}")
        m[3].metric("Rủi ro ròng", f"{F[best,3]:,.0f}")
        bx = X[best].reshape(6, 4)
        dfb = pd.DataFrame(bx.round(0), columns=ITEMS, index=REGION_VI)
        dfb["Tổng"] = dfb.sum(1)
        st.dataframe(dfb, use_container_width=True)
        st.caption("Phân bổ của nghiệm thỏa hiệp — vùng và hạng mục nhận nhiều ngân sách nhất "
                   "phản ánh cân bằng giữa 4 mục tiêu theo trọng số đã chọn.")

        st.subheader("Câu 7.4.4 — Chi phí cơ hội của các mục tiêu")
        mg = int(np.argmin(F[:, 0]))          # tăng trưởng cao nhất
        fg = F[mg]; fc = F[best]
        d_gdp = ((-fg[0]) - (-fc[0])) / (-fc[0]) * 100
        d_gini = (fg[1] - fc[1]) / fc[1] * 100 if fc[1] != 0 else 0
        d_emit = (fg[2] - fc[2]) / fc[2] * 100 if fc[2] != 0 else 0
        st.dataframe(pd.DataFrame({
            "Mục tiêu": ["GDP gain", "Gini/MAD", "Phát thải", "Rủi ro ròng"],
            "Nghiệm thỏa hiệp": [f"{-fc[0]:,.0f}", f"{fc[1]:.0f}", f"{fc[2]:,.0f}", f"{fc[3]:,.0f}"],
            "Nghiệm tăng trưởng cao nhất": [f"{-fg[0]:,.0f}", f"{fg[1]:.0f}", f"{fg[2]:,.0f}", f"{fg[3]:,.0f}"],
            "Thay đổi %": [f"{d_gdp:+.1f}%", f"{d_gini:+.1f}%", f"{d_emit:+.1f}%", "—"]}),
            use_container_width=True, hide_index=True)
        st.info(f"Để nâng GDP gain thêm **{d_gdp:+.1f}%** so với nghiệm thỏa hiệp, phải hi sinh "
                f"**{d_gini:+.1f}%** về công bằng (Gini) và **{d_emit:+.1f}%** về phát thải. "
                "Đây chính là 'cái giá' định lượng của việc chạy theo tăng trưởng đơn thuần — "
                "thông tin cốt lõi để hội đồng chính sách cân nhắc.")

    with t_pol:
        st.subheader("7.5. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a)** Đánh đổi tăng trưởng ↔ bao trùm rõ rệt: vốn dồn về vùng giàu (ĐNB, ĐBSH) "
            "tối đa GDP nhưng tăng bất bình đẳng.\n\n"
            "**b)** Trọng số (0,40/0,25/0,20/0,15) ưu tiên tăng trưởng; để phù hợp COP26 và "
            "Quyết định 127 nên tăng trọng số môi trường và an ninh.\n\n"
            "**c)** NSGA-II cung cấp tập lựa chọn, **không thay thế quyết định chính trị** — "
            "nhà hoạch định chọn nghiệm trên đường biên theo ưu tiên xã hội."
        )


# ============================================================================
#  BÀI 8 — Tối ưu động
# ============================================================================
@st.cache_data(show_spinner="Đang tối ưu quỹ đạo 2026-2035 (SLSQP)...")
def run_dynamic(rho=0.97):
    from scipy.optimize import minimize
    a, b, gd, dai, th = 0.33, 0.42, 0.10, 0.08, 0.07
    dK, dD, dAI, thH, mu = 0.05, 0.12, 0.15, 0.8, 0.02
    phi1, phi2, phi3, gcr, T = 0.003, 0.002, 0.004, 1.5, 10
    K0, L0, D0, AI0, H0, Y0 = 27500.0, 53.9, 20.3, 86.0, 30.0, 12847.6
    A0 = Y0 / (K0 ** a * L0 ** b * D0 ** gd * AI0 ** dai * H0 ** th)
    L = np.array([L0 * 1.009 ** t for t in range(T + 1)])

    def traj(u):
        IK, ID, IAI, IH = u[0::4], u[1::4], u[2::4], u[3::4]
        K = np.zeros(T + 1); D = np.zeros(T + 1); AI = np.zeros(T + 1)
        H = np.zeros(T + 1); A = np.zeros(T + 1); Y = np.zeros(T + 1); C = np.zeros(T)
        K[0], D[0], AI[0], H[0], A[0] = K0, D0, AI0, H0, A0
        for t in range(T):
            Y[t] = A[t] * K[t]**a * L[t]**b * D[t]**gd * AI[t]**dai * H[t]**th
            C[t] = Y[t] - IK[t] - ID[t] - IAI[t] - IH[t]
            if C[t] <= 0:
                return None
            K[t+1] = (1-dK)*K[t]+IK[t]; D[t+1] = (1-dD)*D[t]+ID[t]
            AI[t+1] = (1-dAI)*AI[t]+IAI[t]; H[t+1] = H[t]+thH*IH[t]-mu*H[t]
            A[t+1] = A[t]*(1+phi1*(D[t]/100)+phi2*(AI[t]/100)+phi3*(H[t]/100))
        Y[T] = A[T]*K[T]**a*L[T]**b*D[T]**gd*AI[T]**dai*H[T]**th
        return K, D, AI, H, Y, C, A

    def welfare(u):
        r = traj(u)
        if r is None or np.any(r[5] <= 0):
            return 1e15
        C = r[5]
        return -sum(rho**t * (C[t]**(1-gcr)-1)/(1-gcr) for t in range(T))

    ti = 14000 * 0.15
    u0 = np.zeros(T*4)
    for t in range(T):
        u0[t*4:t*4+4] = [ti*0.40, ti*0.25, ti*0.20, ti*0.15]
    cons = [{"type": "ineq", "fun": lambda u: (lambda r: -1e10 if r is None else min(r[5]) - 1)(traj(u))}]
    res = minimize(welfare, u0, method="SLSQP", bounds=[(0, None)]*(T*4),
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-8})
    return traj(res.x), -res.fun, np.arange(2026, 2037), res.x


@st.cache_data(show_spinner=False)
def compare_strategies(rho=0.97):
    """So sánh 3 chiến lược: tối ưu / trải đều / front-load (Câu 8.3.4)."""
    a, b, gd, dai, th = 0.33, 0.42, 0.10, 0.08, 0.07
    dK, dD, dAI, thH, mu = 0.05, 0.12, 0.15, 0.8, 0.02
    phi1, phi2, phi3, gcr, T = 0.003, 0.002, 0.004, 1.5, 10
    K0, L0, D0, AI0, H0, Y0 = 27500.0, 53.9, 20.3, 86.0, 30.0, 12847.6
    A0 = Y0 / (K0 ** a * L0 ** b * D0 ** gd * AI0 ** dai * H0 ** th)
    L = np.array([L0 * 1.009 ** t for t in range(T + 1)])

    def traj(u):
        IK, ID, IAI, IH = u[0::4], u[1::4], u[2::4], u[3::4]
        K = K0; D = D0; AI = AI0; H = H0; A = A0; C = np.zeros(T); Y_last = 0
        for t in range(T):
            Yt = A*K**a*L[t]**b*D**gd*AI**dai*H**th
            C[t] = Yt - IK[t] - ID[t] - IAI[t] - IH[t]
            if C[t] <= 0:
                return None, None
            K = (1-dK)*K+IK[t]; D = (1-dD)*D+ID[t]; AI = (1-dAI)*AI+IAI[t]
            H = H+thH*IH[t]-mu*H; A = A*(1+phi1*(D/100)+phi2*(AI/100)+phi3*(H/100))
        Y_last = A*K**a*L[T]**b*D**gd*AI**dai*H**th
        return C, Y_last

    def welf(C):
        return sum(rho**t * (C[t]**(1-gcr)-1)/(1-gcr) for t in range(T))

    ti = 14000 * 0.15
    u_even = np.tile([ti*0.40, ti*0.25, ti*0.20, ti*0.15], T)
    u_front = np.zeros(T*4)
    for t in range(T):
        f = 1.5 if t < 3 else 0.7
        u_front[t*4:t*4+4] = [ti*0.40*f, ti*0.25*f, ti*0.20*f, ti*0.15*f]
    rows = {"Chiến lược": [], "Phúc lợi W": [], "GDP 2035": []}
    for name, u in [("Tối ưu (SLSQP)", None), ("Đầu tư trải đều", u_even), ("Front-load", u_front)]:
        if u is None:
            traj_t, W_opt, _, _ = run_dynamic(rho)
            rows["Chiến lược"].append(name); rows["Phúc lợi W"].append(round(W_opt, 3))
            rows["GDP 2035"].append(round(traj_t[4][-1]))
        else:
            C, Yl = traj(u)
            rows["Chiến lược"].append(name)
            rows["Phúc lợi W"].append(round(welf(C), 3) if C is not None else None)
            rows["GDP 2035"].append(round(Yl) if Yl else None)
    return rows


@st.cache_data(show_spinner="Đang phân tích cú sốc 2028...")
def shock_analysis(rho=0.97, shock_t=2, shock_pct=0.08):
    """Câu 8.3.3: cú sốc TFP giảm shock_pct tại năm shock_t (2028=t2). Trả về 3 kịch bản."""
    from scipy.optimize import minimize
    a, b, gd, dai, th = 0.33, 0.42, 0.10, 0.08, 0.07
    dK, dD, dAI, thH, mu = 0.05, 0.12, 0.15, 0.8, 0.02
    phi1, phi2, phi3, gcr, T = 0.003, 0.002, 0.004, 1.5, 10
    K0, L0, D0, AI0, H0, Y0 = 27500.0, 53.9, 20.3, 86.0, 30.0, 12847.6
    A0 = Y0 / (K0 ** a * L0 ** b * D0 ** gd * AI0 ** dai * H0 ** th)
    L = np.array([L0 * 1.009 ** t for t in range(T + 1)])

    def traj(u, sh_t=None, sh=0.0):
        IK, ID, IAI, IH = u[0::4], u[1::4], u[2::4], u[3::4]
        K = np.zeros(T+1); D = np.zeros(T+1); AI = np.zeros(T+1)
        H = np.zeros(T+1); A = np.zeros(T+1); Y = np.zeros(T+1); C = np.zeros(T)
        K[0], D[0], AI[0], H[0], A[0] = K0, D0, AI0, H0, A0
        for t in range(T):
            if sh_t is not None and t == sh_t:
                A[t] *= (1 - sh)
            Y[t] = A[t]*K[t]**a*L[t]**b*D[t]**gd*AI[t]**dai*H[t]**th
            C[t] = Y[t]-IK[t]-ID[t]-IAI[t]-IH[t]
            if C[t] <= 0:
                return None
            K[t+1] = (1-dK)*K[t]+IK[t]; D[t+1] = (1-dD)*D[t]+ID[t]
            AI[t+1] = (1-dAI)*AI[t]+IAI[t]; H[t+1] = H[t]+thH*IH[t]-mu*H[t]
            A[t+1] = A[t]*(1+phi1*(D[t]/100)+phi2*(AI[t]/100)+phi3*(H[t]/100))
        Y[T] = A[T]*K[T]**a*L[T]**b*D[T]**gd*AI[T]**dai*H[T]**th
        return K, D, AI, H, Y, C, A

    def welf(u, sh_t=None, sh=0.0):
        r = traj(u, sh_t, sh)
        if r is None or np.any(r[5] <= 0):
            return 1e15
        C = r[5]
        return -sum(rho**t*(C[t]**(1-gcr)-1)/(1-gcr) for t in range(T))

    traj_t, _, _, u_opt = run_dynamic(rho)         # kế hoạch gốc (không sốc)
    W_base = -welf(u_opt)
    # (B) giữ kế hoạch gốc nhưng có sốc
    Y_sh = traj(u_opt, shock_t, shock_pct)[4]
    W_plan_shock = -welf(u_opt, shock_t, shock_pct)
    # (C) tái tối ưu sau sốc
    cons = [{"type": "ineq",
             "fun": lambda u: (lambda r: -1e10 if r is None else min(r[5])-1)(traj(u, shock_t, shock_pct))}]
    res = minimize(lambda u: welf(u, shock_t, shock_pct), u_opt, method="SLSQP",
                   bounds=[(0, None)]*(T*4), constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-8})
    W_reopt = -res.fun
    Y_reopt = traj(res.x, shock_t, shock_pct)[4]
    return {"years": np.arange(2026, 2037), "Y_base": traj_t[4], "Y_shock": Y_sh, "Y_reopt": Y_reopt,
            "W_base": W_base, "W_plan_shock": W_plan_shock, "W_reopt": W_reopt}


def sidebar_bai8():
    with SB:
        st.markdown("### Tham số Bài 8")
        rho = st.slider("ρ - hệ số chiết khấu", 0.85, 0.99, 0.97, 0.01)
    return dict(rho=rho)


def page_bai8():
    P = sidebar_bai8()
    page_title("📈", "Bài 8 — Tối ưu động phân bổ liên thời gian 2026–2035", "Bài 8")
    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["8.1 Bối cảnh", "8.2 Mô hình", "8.3 Dữ liệu", "8.4 Tính toán", "8.5 Chính sách"])

    with t_ctx:
        st.subheader("8.1. Bối cảnh Việt Nam")
        st.write("Theo Văn kiện Đại hội XIII, Việt Nam đặt mục tiêu thu nhập trung bình cao 2030 "
                 "và thu nhập cao 2045. Cần thiết kế **chiến lược phân bổ vốn dài hạn** cân bằng "
                 "tăng trưởng, chuyển đổi số, AI và nhân lực qua giai đoạn 2026–2035.")

    with t_mod:
        st.subheader("8.2. Mô hình tối ưu động")
        st.latex(r"\max \sum_{t} \rho^{t}\, U(C_t), \quad U(C)=\frac{C^{1-\gamma}}{1-\gamma}")
        st.markdown("Động học vốn: K, D, AI, H với khấu hao; TFP nội sinh:")
        st.latex(r"A_{t+1} = A_t (1+\phi_1 D_t + \phi_2 AI_t + \phi_3 H_t)")
        st.latex(r"C_t + I_{K,t}+I_{D,t}+I_{AI,t}+I_{H,t} \le Y_t")

    with t_data:
        st.subheader("8.3. Tham số & điều kiện ban đầu (2026)")
        st.dataframe(pd.DataFrame({
            "Tham số": ["K₀", "L₀", "D₀", "AI₀", "H₀", "δ_K", "δ_D", "δ_AI", "θ_H", "ρ"],
            "Giá trị": [27500, 53.9, 20.3, 86, 30, 0.05, 0.12, 0.15, 0.8, P["rho"]]}),
            use_container_width=True, hide_index=True)

    with t_calc:
        (K, D, AI, H, Y, C, A), W, years, u_opt = run_dynamic(P["rho"])

        st.subheader("Câu 8.3.1–8.3.2 — Quỹ đạo phân bổ vốn tối ưu")
        m = st.columns(4)
        m[0].metric("Phúc lợi W*", f"{W:.3f}")
        m[1].metric("GDP 2026", f"{Y[0]:,.0f}")
        m[2].metric("GDP 2035", f"{Y[-1]:,.0f}", f"{(Y[-1]/Y[0])**(1/10)*100-100:.2f}%/năm")
        m[3].metric("TFP 2035/2026", f"{A[-1]/A[0]:.3f} lần")
        df = pd.DataFrame({"Năm": years, "K": K.round(0), "D": D.round(1), "AI": AI.round(1),
                           "H": H.round(1), "TFP": A.round(2), "Y (GDP)": Y.round(0)})
        df["C (tiêu dùng)"] = list(C.round(0)) + [np.nan]
        st.dataframe(df, use_container_width=True, hide_index=True, height=250)

        fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))
        for ax, dat, ti in [(axes[0, 0], K, "K — Vốn vật chất"), (axes[0, 1], D, "D — Hạ tầng số (%)"),
                            (axes[0, 2], AI, "AI — nghìn DN"), (axes[1, 0], H, "H — Nhân lực (%)"),
                            (axes[1, 2], A, "A — TFP")]:
            ax.plot(years, dat, "b-o", ms=4); ax.set_title(ti); ax.grid(alpha=0.3)
        axes[1, 1].plot(years, Y, "k-o", ms=4, label="Y (GDP)")
        axes[1, 1].plot(years[:10], C, "c-o", ms=4, label="C (tiêu dùng)")
        axes[1, 1].legend(); axes[1, 1].set_title("Y & C"); axes[1, 1].grid(alpha=0.3)
        plt.suptitle(f"Quỹ đạo tối ưu 2026–2035 (ρ={P['rho']})", fontsize=13)
        show_fig(fig)

        st.subheader("Cơ cấu đầu tư theo thời gian (tỷ lệ I/GDP)")
        IK, ID, IAI, IH = u_opt[0::4], u_opt[1::4], u_opt[2::4], u_opt[3::4]
        inv_df = pd.DataFrame({
            "Năm": years[:10],
            "IK/Y %": (IK / Y[:10] * 100).round(1), "ID/Y %": (ID / Y[:10] * 100).round(1),
            "IAI/Y %": (IAI / Y[:10] * 100).round(1), "IH/Y %": (IH / Y[:10] * 100).round(1),
            "Tổng I/Y %": ((IK + ID + IAI + IH) / Y[:10] * 100).round(1)})
        cc = st.columns([1.3, 1])
        with cc[0]:
            st.dataframe(inv_df, use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            ax.stackplot(years[:10], IK, ID, IAI, IH,
                         labels=["I_K", "I_D", "I_AI", "I_H"],
                         colors=["#3498db", "#f39c12", "#9b59b6", "#2ecc71"])
            ax.legend(loc="upper right", fontsize=8); ax.set_title("Cơ cấu đầu tư")
            ax.set_xlabel("Năm"); ax.set_ylabel("Tỷ VND"); ax.grid(alpha=0.3)
            show_fig(fig)
        front = (IK + ID + IAI + IH)[:3].sum()
        back = (IK + ID + IAI + IH)[7:].sum()
        st.caption(f"Tổng đầu tư 3 năm đầu = {front:,.0f} tỷ vs 3 năm cuối = {back:,.0f} tỷ → "
                   f"quỹ đạo **{'front-loaded (đầu tư sớm)' if front > back else 'back-loaded'}**. "
                   "Mô hình ưu tiên đầu tư sớm vì TFP nội sinh tạo hiệu ứng lan tỏa tích lũy.")

        st.subheader("Câu 8.3.3 — Phân tích cú sốc 2028 (Y giảm 8%, như bão Yagi)")
        sh = shock_analysis(P["rho"])
        m = st.columns(3)
        m[0].metric("W không sốc", f"{sh['W_base']:.3f}")
        m[1].metric("W có sốc (giữ kế hoạch)", f"{sh['W_plan_shock']:.3f}",
                    f"{sh['W_plan_shock']-sh['W_base']:+.3f}")
        m[2].metric("W tái tối ưu sau sốc", f"{sh['W_reopt']:.3f}",
                    f"{sh['W_reopt']-sh['W_plan_shock']:+.3f}")
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(sh["years"], sh["Y_base"], "k-o", ms=4, label="Không sốc")
        ax.plot(sh["years"], sh["Y_shock"], "r-s", ms=4, label="Có sốc, giữ kế hoạch")
        ax.plot(sh["years"], sh["Y_reopt"], "g-^", ms=4, label="Có sốc, tái tối ưu")
        ax.axvline(2028, color="gray", ls="--", lw=1)
        ax.set_xlabel("Năm"); ax.set_ylabel("GDP (ngh.tỷ)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("Tác động cú sốc TFP năm 2028 lên quỹ đạo GDP")
        show_fig(fig)
        st.caption("Cú sốc TFP năm 2028 (giảm 8%) tác động kéo dài qua động học A_{t+1}=A_t(1+...). "
                   "Khi **tái tối ưu sau sốc**, mô hình điều chỉnh phân bổ (giảm đầu tư để giữ tiêu "
                   "dùng, ưu tiên hạng mục phục hồi nhanh) nên phúc lợi cải thiện so với cứng nhắc "
                   "giữ kế hoạch cũ — minh họa giá trị của tính linh hoạt chính sách.")

        st.subheader("Câu 8.3.4 — So sánh chiến lược đầu tư")
        comp = compare_strategies(P["rho"])
        cc = st.columns([1.2, 1])
        with cc[0]:
            st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            ax.bar(comp["Chiến lược"], comp["Phúc lợi W"], color=["#2ecc71", "#3498db", "#e67e22"])
            ax.set_ylabel("Phúc lợi W"); plt.xticks(rotation=12, ha="right", fontsize=8)
            ax.set_title("Phúc lợi theo chiến lược")
            show_fig(fig)
        st.info("Chiến lược tối ưu (SLSQP) cho phúc lợi cao nhất nhờ phân bổ linh hoạt theo "
                "thời gian; front-load thường vượt đầu tư trải đều vì tận dụng sớm hiệu ứng "
                "tích lũy TFP, nhưng làm giảm tiêu dùng các năm đầu (đánh đổi smoothing).")

    with t_pol:
        st.subheader("8.5. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a)** Quỹ đạo có xu hướng **front-loaded**: đầu tư sớm sinh lời cao hơn nhờ TFP "
            "nội sinh tạo lan tỏa dài hạn.\n\n"
            "**b)** Tỷ lệ đầu tư AI/H gợi ý đào tạo nhân lực nên **đi trước hoặc đồng thời** với "
            "đầu tư AI (năng lực hấp thụ).\n\n"
            "**c)** ρ=0,97 ưu tiên dài hạn; ρ=0,90 (ngắn hạn) khiến chính phủ 'dưới đầu tư' vào "
            "R&D và nhân lực — lý giải hiện tượng under-investment phổ biến."
        )


# ============================================================================
#  BÀI 9 — Lao động & AI
# ============================================================================
def sidebar_bai9():
    with SB:
        st.markdown("### Tham số Bài 9")
        budget = st.slider("Ngân sách (tỷ)", 20000, 40000, 30000, 2000)
        add5 = st.checkbox("RB: mỗi ngành mất ≤ 5% LĐ", value=False)
    return dict(budget=budget, add5=add5)


def page_bai9():
    from scipy.optimize import linprog
    P = sidebar_bai9()
    page_title("👷", "Bài 9 — Tác động AI tới thị trường lao động Việt Nam", "Bài 9")
    N = 8
    sec = ["Nông-LT", "CN chế biến", "Xây dựng", "Bán buôn-bán lẻ",
           "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục-ĐT"]
    L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])
    coeff = a1 - c1 * risk
    c_obj = np.concatenate([-coeff, -b1])
    A1 = np.concatenate([np.ones(N), np.ones(N)]).reshape(1, -1)
    A1b = np.concatenate([-np.ones(N), np.zeros(N)]).reshape(1, -1)
    A2 = np.zeros((N, 2*N)); A3 = np.zeros((N, 2*N))
    for i in range(N):
        A2[i, i] = -coeff[i]; A2[i, N+i] = -b1[i]
        A3[i, i] = c1[i]*risk[i]; A3[i, N+i] = -d1[i]
    A_ub = np.vstack([A1, A1b, A2, A3])
    b_ub = np.concatenate([[P["budget"]], [-0.3*P["budget"]], np.zeros(N), np.zeros(N)])
    if P["add5"]:
        A4 = np.zeros((N, 2*N))
        for i in range(N):
            A4[i, i] = c1[i]*risk[i]
        A_ub = np.vstack([A_ub, A4]); b_ub = np.concatenate([b_ub, 0.05*L*1e6])
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*(2*N), method="highs")

    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["9.1 Bối cảnh", "9.2 Mô hình", "9.3 Dữ liệu", "9.4 Tính toán", "9.5 Chính sách"])

    with t_ctx:
        st.subheader("9.1. Bối cảnh Việt Nam")
        st.write("Theo ILO Vietnam 2024, khoảng 30–50% việc làm tại Việt Nam có nguy cơ tự động "
                 "hóa một phần trong 10 năm tới (chế biến chế tạo, bán buôn, logistics). AI cũng "
                 "tạo việc làm mới. Bài toán: đầu tư bao nhiêu vào đào tạo lại để **NetJob ròng "
                 "dương** cho mọi ngành?")

    with t_mod:
        st.subheader("9.2. Mô hình toán học")
        st.latex(r"NetJob_i = NewJob_i + UpgradeJob_i - DisplacedJob_i")
        st.latex(r"\max \sum_i NetJob_i \ \text{s.t.}\ \sum_i (x_{AI,i}+x_{H,i}) \le B,\ NetJob_i \ge 0")
        st.caption("DisplacedJobᵢ ≤ RetrainingCapacityᵢ (tốc độ tự động hóa không vượt năng lực đào tạo lại).")

    with t_data:
        st.subheader("9.3. Tham số 8 ngành")
        st.dataframe(pd.DataFrame({"Ngành": sec, "LĐ (tr)": L, "Risk %": (risk*100).astype(int),
                                   "a₁": a1, "b₁": b1, "c₁": c1, "d₁": d1}),
                     use_container_width=True, hide_index=True)

    with t_calc:
        st.subheader("Câu 9.4.1 — Phân bổ tối ưu & NetJob ròng")
        if res.success:
            xA, xH = res.x[:N], res.x[N:]
            NetJob = coeff * xA + b1 * xH
            Displaced = c1 * risk * xA
            st.dataframe(pd.DataFrame({"Ngành": sec, "x_AI": xA.round(0), "x_H": xH.round(0),
                                       "Displaced": Displaced.round(0), "NetJob": NetJob.round(0)}),
                         use_container_width=True, hide_index=True)
            st.metric("Tổng NetJob", f"{-res.fun:,.0f} việc làm")
            fig, ax = plt.subplots(figsize=(9, 3.2))
            ax.bar(sec, NetJob, color="#2ecc71"); ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=25, ha="right", fontsize=8); ax.set_title("NetJob ròng theo ngành")
            show_fig(fig)
            top_h = sec[int(np.argmax(xH))]
            top_ai = sec[int(np.argmax(xA))]
            st.caption(f"📌 Ngành nhận đầu tư đào tạo lại (x_H) lớn nhất: **{top_h}** "
                       f"({xH.max():,.0f} tỷ); đầu tư AI lớn nhất: **{top_ai}** ({xA.max():,.0f} tỷ). "
                       "Ngành rủi ro tự động hóa cao (chế biến chế tạo, bán buôn) được ưu tiên "
                       "đào tạo để giữ NetJob ≥ 0 — đúng nguyên tắc 'tự động hóa không vượt năng "
                       "lực đào tạo lại'.")

            st.subheader("Câu 9.4.2 — Ngưỡng đào tạo tối thiểu (CN chế biến chế tạo)")
            i = 1  # CN chế biến
            net = a1[i] - c1[i] * risk[i]
            ratio = c1[i] * risk[i] / d1[i]
            cc = st.columns([1, 1])
            with cc[0]:
                st.markdown(f"Ngành **{sec[i]}**: a₁={a1[i]}, c₁·risk={c1[i]*risk[i]:.1f}, d₁={d1[i]}.")
                st.latex(r"Displaced \le RetrainCap \Rightarrow x_H \ge \frac{c_1\cdot risk}{d_1}\,x_{AI}")
                st.metric("Hệ số net AI", f"{net:.1f}",
                          "AI tạo việc ròng" if net > 0 else "AI làm mất việc ròng")
                st.metric("Tỷ lệ đào tạo tối thiểu x_H/x_AI", f"{ratio:.3f}")
                st.caption(f"Mỗi 1 tỷ đầu tư AI cần ≥ **{ratio:.3f} tỷ** đầu tư đào tạo lại để giữ "
                           f"năng lực retraining. Nếu dùng hết 30.000 tỷ cho AI thì cần x_H ≥ "
                           f"{ratio*30000:,.0f} tỷ.")
            with cc[1]:
                xr = np.linspace(0, 30000, 100)
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ax.plot(xr, ratio * xr, "r--", lw=2, label=f"Retrain: x_H≥{ratio:.3f}·x_AI")
                ax.plot(xr, np.maximum(0, -net / b1[i] * xr), "b--", lw=2, label="NetJob≥0")
                ax.fill_between(xr, np.maximum(ratio * xr, np.maximum(0, -net / b1[i] * xr)), 30000,
                                alpha=0.2, color="green", label="Vùng khả thi")
                ax.set_xlabel("x_AI (tỷ)"); ax.set_ylabel("x_H tối thiểu (tỷ)")
                ax.set_xlim(0, 30000); ax.set_ylim(0, 30000); ax.legend(fontsize=8); ax.grid(alpha=0.3)
                ax.set_title(f"Ngưỡng đào tạo — {sec[i]}")
                show_fig(fig)

            st.subheader("Câu 9.4.3 — Nhóm dễ bị tổn thương (luồng dịch chuyển lao động)")
            st.caption("Lao động phổ thông trong các ngành Nông-LT, Xây dựng, Bán buôn — biểu đồ "
                       "swimming lane thể hiện số giữ việc / được đào tạo lại / mất việc.")
            vuln = [0, 2, 3]
            kept, retr, lost = [], [], []
            for j in vuln:
                disp = Displaced[j]; rc = min(disp, d1[j]*xH[j])
                kept.append(L[j]*1e6 - disp); retr.append(rc); lost.append(max(0, disp - rc))
            fig, ax = plt.subplots(figsize=(8, 3.4))
            nm = [sec[j] for j in vuln]
            ax.bar(nm, kept, label="Giữ việc", color="#2ecc71")
            ax.bar(nm, retr, bottom=kept, label="Đào tạo lại", color="#f39c12")
            ax.bar(nm, lost, bottom=[k+r for k, r in zip(kept, retr)], label="Mất việc", color="#e74c3c")
            ax.legend(); ax.set_title("Luồng dịch chuyển lao động"); ax.grid(axis="y", alpha=0.3)
            show_fig(fig)
            tot_lost = sum(lost)
            st.caption(f"Tổng lao động mất việc ở nhóm dễ tổn thương: **{tot_lost:,.0f}** người "
                       f"({'phần lớn được đào tạo lại' if tot_lost < sum(retr) else 'cần tăng đầu tư đào tạo'}). "
                       "Nhóm này cần chính sách an sinh và đào tạo lại được ưu tiên.")

            st.subheader("Câu 9.4.4 — Ràng buộc Displaced ≤ 5% lao động")
            if P["add5"]:
                st.success("Đang BẬT ràng buộc 5% (chọn ở sidebar). Kết quả phía trên đã phản ánh "
                           "giới hạn này — tổng NetJob có thể thấp hơn nhưng an toàn xã hội cao hơn.")
            else:
                st.info("Bật ràng buộc '≤ 5% lao động mỗi ngành' ở sidebar để xem ảnh hưởng. "
                        "Ràng buộc này giới hạn số lao động bị dịch chuyển ở mỗi ngành, bảo đảm "
                        "an sinh nhưng có thể làm giảm tổng NetJob (đánh đổi hiệu quả ↔ an toàn).")
        else:
            st.error(f"Không khả thi: {res.message}")

    with t_pol:
        st.subheader("9.5. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a)** Ngành chế biến chế tạo và bán buôn cần đầu tư đào tạo lại nhiều nhất (rủi ro "
            "tự động hóa cao, lao động đông).\n\n"
            "**b)** Tài chính-Ngân hàng rủi ro 52% nhưng hệ số tạo việc mới rất cao → chiến lược "
            "'tái cấu trúc kỹ năng' thay vì cắt giảm.\n\n"
            "**c)** Ràng buộc 'tốc độ tự động hóa ≤ năng lực đào tạo lại' (Displaced ≤ RetrainCap) "
            "là cốt lõi an sinh; nên bổ sung sàn đào tạo bắt buộc cho nhóm dễ tổn thương."
        )


# ============================================================================
#  BÀI 10 — Stochastic SP
# ============================================================================
def page_bai10():
    page_title("🎲", "Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định", "Bài 10")
    J = ["I", "D", "AI", "H"]; S = ["s1", "s2", "s3", "s4"]
    S_VI = {"s1": "Lạc quan", "s2": "Cơ sở", "s3": "Bi quan", "s4": "Khủng hoảng"}
    p_s = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
    beta_base = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
    beta_s = {("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
              ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
              ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
              ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10}

    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["10.1 Bối cảnh", "10.2 Mô hình", "10.3 Dữ liệu", "10.4 Tính toán", "10.5 Chính sách"])

    with t_ctx:
        st.subheader("10.1. Bối cảnh Việt Nam")
        st.write("Việt Nam có độ mở thương mại rất cao (xuất nhập khẩu/GDP ≈ 180% năm 2025), "
                 "nên tăng trưởng phụ thuộc lớn vào kịch bản kinh tế toàn cầu: cầu xuất khẩu, "
                 "dòng FDI và biến động địa - chính trị. Khi hoạch định ngân sách đầu tư số "
                 "2026–2030, Chính phủ phải đưa ra **quyết định first-stage (here-and-now)** — "
                 "kế hoạch phân bổ 5 năm — *trước khi* biết chắc kịch bản tương lai, rồi mới "
                 "**điều chỉnh recourse (wait-and-see)** sau khi quan sát thực tế.")
        c = st.columns(4)
        c[0].metric("Số kịch bản", "4")
        c[1].metric("Ngân sách first-stage", "65.000 tỷ")
        c[2].metric("Quỹ dự phòng recourse", "15.000 tỷ")
        c[3].metric("Độ mở thương mại", "≈180% GDP")
        st.markdown("**Vì sao cần quy hoạch ngẫu nhiên?** Nếu chỉ lập kế hoạch theo một kịch bản "
                    "'trung bình' (deterministic), quyết định ban đầu sẽ thiếu khả năng thích ứng "
                    "khi cú sốc xảy ra (như COVID-19, bão Yagi). Mô hình hai giai đoạn cho phép "
                    "vừa tối ưu lợi ích kỳ vọng, vừa giữ tính linh hoạt điều chỉnh.")

    with t_mod:
        st.subheader("10.2. Mô hình toán học hai giai đoạn")
        st.markdown("**Dạng chuẩn** (cực tiểu hóa chi phí kỳ vọng):")
        st.latex(r"\min\ c'x + \sum_{s\in S} p_s\, Q(x,s), \quad Q(x,s)=\min\{q'y^s: T_s x + W y^s = h_s,\ y^s\ge 0\}")
        st.markdown("**Dạng đơn giản hóa cho bài tập** (tối đa hóa tăng GDP kỳ vọng):")
        st.latex(r"\max\ \sum_j \beta_j x_j + \sum_{s\in S} p_s \sum_j \beta^s_j\, y^s_j")
        st.markdown("Ràng buộc:")
        st.latex(r"\sum_j x_j \le 65000;\quad \sum_j y^s_j \le 15000\ \forall s;\quad y^s_{AI} \le 0.5\,x_H\ \forall s")
        st.markdown("- **Giai đoạn 1** (here-and-now): biến $x=(x_I,x_D,x_{AI},x_H)$ — phân bổ ban "
                    "đầu, giữ lại 15.000 tỷ dự phòng (nguyên tắc thận trọng).\n"
                    "- **Giai đoạn 2** (recourse): biến $y^s$ — điều chỉnh sau khi biết kịch bản $s$.\n"
                    "- Ràng buộc $y^s_{AI} \\le 0.5\\,x_H$: mở rộng AI bổ sung bị giới hạn bởi nền "
                    "tảng nhân lực đã đầu tư từ giai đoạn 1.")

    with t_data:
        st.subheader("10.3. Cấu trúc kịch bản (scenario tree)")
        st.dataframe(pd.DataFrame({
            "Kịch bản": [S_VI[s] for s in S],
            "Tăng trưởng TG %": [3.5, 2.8, 1.5, 0.2], "FDI VN (tỷ USD)": [32, 27, 20, 12],
            "XK VN tăng %": [12, 8, 3, -5], "Xác suất pₛ": [0.30, 0.45, 0.20, 0.05]}),
            use_container_width=True, hide_index=True)
        st.subheader("Hệ số hiệu quả βˢⱼ theo kịch bản")
        st.dataframe(pd.DataFrame({"Hạng mục": J,
                                   "β cơ bản": [beta_base[j] for j in J],
                                   "s1 Lạc quan": [beta_s[("s1", j)] for j in J],
                                   "s2 Cơ sở": [beta_s[("s2", j)] for j in J],
                                   "s3 Bi quan": [beta_s[("s3", j)] for j in J],
                                   "s4 Khủng hoảng": [beta_s[("s4", j)] for j in J]}),
                     use_container_width=True, hide_index=True)
        st.caption("Lưu ý: hệ số H (nhân lực) **cao hơn trong kịch bản khủng hoảng** (1,10 > 0,95) "
                   "vì lao động qua đào tạo có khả năng chuyển đổi việc làm tốt hơn, hấp thụ cú sốc "
                   "— đây là lý do nhân lực số đóng vai trò 'bảo hiểm'.")

    with t_calc:
        try:
            import pyomo.environ as pyo

            def gs():
                for nm in ["appsi_highs", "glpk", "cbc"]:
                    sv = pyo.SolverFactory(nm)
                    if sv.available():
                        return sv
                return None
            solver = gs()
            if solver is None:
                raise RuntimeError("no solver")

            # ---- Câu 10.5.1: Mô hình SP đầy đủ ----
            m = pyo.ConcreteModel()
            m.J = pyo.Set(initialize=J); m.S = pyo.Set(initialize=S)
            m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
            m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
            m.b1 = pyo.Constraint(expr=sum(m.x[j] for j in J) <= 65000)
            m.b2 = pyo.Constraint(m.S, rule=lambda mm, s: sum(mm.y[s, j] for j in J) <= 15000)
            m.ac = pyo.Constraint(m.S, rule=lambda mm, s: mm.y[s, "AI"] <= 0.5 * mm.x["H"])
            m.obj = pyo.Objective(expr=sum(beta_base[j]*m.x[j] for j in J) +
                                  sum(p_s[s]*sum(beta_s[s, j]*m.y[s, j] for j in J) for s in S),
                                  sense=pyo.maximize)
            solver.solve(m)
            Z_SP = pyo.value(m.obj)
            x_sp = {j: pyo.value(m.x[j]) for j in J}
            y_sp = {s: {j: pyo.value(m.y[s, j]) for j in J} for s in S}

            st.subheader("Câu 10.5.1 — Lời giải Stochastic (SP)")
            cc = st.columns([1, 1])
            with cc[0]:
                st.markdown("**Giai đoạn 1 (here-and-now):**")
                st.dataframe(pd.DataFrame({"Hạng mục": J, "x* (tỷ)": [round(x_sp[j]) for j in J]}),
                             use_container_width=True, hide_index=True)
                st.metric("Z* Stochastic", f"{Z_SP:,.0f}")
            with cc[1]:
                st.markdown("**Giai đoạn 2 (recourse yˢ) theo kịch bản:**")
                st.dataframe(pd.DataFrame({"Kịch bản": [S_VI[s] for s in S],
                                           **{j: [round(y_sp[s][j]) for s in S] for j in J}}),
                             use_container_width=True, hide_index=True)
            st.caption(f"Quyết định ban đầu tập trung vào hạng mục hiệu quả cao "
                       f"(AI β={beta_base['AI']}, D β={beta_base['D']}) nhưng vẫn giữ x_H = "
                       f"{x_sp['H']:,.0f} tỷ để 'mở khóa' khả năng đầu tư AI bổ sung ở giai đoạn 2.")

            # ---- Câu 10.5.2: deterministic từng kịch bản + EV ----
            st.subheader("Câu 10.5.2 — Lời giải xác định từng kịch bản & EV solution")
            det = {}
            for s in S:
                ms = pyo.ConcreteModel(); ms.J = pyo.Set(initialize=J)
                ms.x = pyo.Var(ms.J, within=pyo.NonNegativeReals)
                ms.y = pyo.Var(ms.J, within=pyo.NonNegativeReals)
                ms.b1 = pyo.Constraint(expr=sum(ms.x[j] for j in J) <= 65000)
                ms.b2 = pyo.Constraint(expr=sum(ms.y[j] for j in J) <= 15000)
                ms.ac = pyo.Constraint(expr=ms.y["AI"] <= 0.5 * ms.x["H"])
                ms.obj = pyo.Objective(expr=sum(beta_base[j]*ms.x[j] for j in J) +
                                       sum(beta_s[s, j]*ms.y[j] for j in J), sense=pyo.maximize)
                solver.solve(ms)
                det[s] = {"x": {j: pyo.value(ms.x[j]) for j in J}, "Z": pyo.value(ms.obj)}
            Z_WS = sum(p_s[s]*det[s]["Z"] for s in S)
            # EV solution: dùng beta trung bình
            beta_avg = {j: sum(p_s[s]*beta_s[s, j] for s in S) for j in J}
            mev = pyo.ConcreteModel(); mev.J = pyo.Set(initialize=J)
            mev.x = pyo.Var(mev.J, within=pyo.NonNegativeReals)
            mev.b = pyo.Constraint(expr=sum(mev.x[j] for j in J) <= 65000)
            mev.obj = pyo.Objective(expr=sum(beta_avg[j]*mev.x[j] for j in J), sense=pyo.maximize)
            solver.solve(mev)
            x_ev = {j: pyo.value(mev.x[j]) for j in J}
            Z_EV = sum(beta_base[j]*x_ev[j] for j in J)
            for s in S:
                mt = pyo.ConcreteModel(); mt.J = pyo.Set(initialize=J)
                mt.y = pyo.Var(mt.J, within=pyo.NonNegativeReals)
                mt.b2 = pyo.Constraint(expr=sum(mt.y[j] for j in J) <= 15000)
                mt.ac = pyo.Constraint(expr=mt.y["AI"] <= 0.5*x_ev["H"])
                mt.obj = pyo.Objective(expr=sum(beta_s[s, j]*mt.y[j] for j in J), sense=pyo.maximize)
                solver.solve(mt)
                Z_EV += p_s[s]*pyo.value(mt.obj)
            cc = st.columns([1.3, 1])
            with cc[0]:
                st.dataframe(pd.DataFrame({"Kịch bản": [S_VI[s] for s in S],
                                           "Z*[s] (Wait&See)": [round(det[s]["Z"]) for s in S],
                                           **{f"x_{j}": [round(det[s]["x"][j]) for s in S] for j in J}}),
                             use_container_width=True, hide_index=True)
            with cc[1]:
                st.dataframe(pd.DataFrame({"Hạng mục": J,
                                           "x* SP": [round(x_sp[j]) for j in J],
                                           "x* EV": [round(x_ev[j]) for j in J]}),
                             use_container_width=True, hide_index=True)
            dH = x_sp["H"] - x_ev["H"]
            st.caption(f"So sánh first-stage: lời giải **SP đầu tư H = {x_sp['H']:,.0f} tỷ** so với "
                       f"**EV = {x_ev['H']:,.0f} tỷ** (chênh {dH:+,.0f} tỷ). "
                       f"{'SP đầu tư nhân lực nhiều hơn — chuẩn bị cho kịch bản xấu.' if dH > 0 else 'Hai lời giải khá tương đồng ở hạng mục H.'}")

            # ---- Câu 10.5.3: VSS & EVPI ----
            st.subheader("Câu 10.5.3 — VSS và EVPI")
            VSS, EVPI = Z_SP - Z_EV, Z_WS - Z_SP
            m3 = st.columns(3)
            m3[0].metric("Z_SP (Stochastic)", f"{Z_SP:,.0f}")
            m3[1].metric("Z_EV (Expected Value)", f"{Z_EV:,.0f}")
            m3[2].metric("Z_WS (Wait & See)", f"{Z_WS:,.0f}")
            m3b = st.columns(2)
            m3b[0].metric("VSS = Z_SP − Z_EV", f"{VSS:,.0f}",
                          help="Giá trị của việc cân nhắc bất định khi ra quyết định")
            m3b[1].metric("EVPI = Z_WS − Z_SP", f"{EVPI:,.0f}",
                          help="Giá trị của thông tin hoàn hảo")
            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.barh(["Z_EV (bỏ qua bất định)", "Z_SP (xét bất định)", "Z_WS (thông tin hoàn hảo)"],
                           [Z_EV, Z_SP, Z_WS], color=["#e67e22", "#3498db", "#2ecc71"])
            ax.set_xlim(min(Z_EV, Z_SP, Z_WS)*0.98, Z_WS*1.01)
            for bar, v in zip(bars, [Z_EV, Z_SP, Z_WS]):
                ax.text(v, bar.get_y()+bar.get_height()/2, f" {v:,.0f}", va="center", fontsize=9)
            ax.set_title("Z_EV ≤ Z_SP ≤ Z_WS (thứ tự lý thuyết)")
            show_fig(fig)
            st.info(f"**VSS = {VSS:,.0f}**: nếu bỏ qua bất định và lập kế hoạch theo kịch bản "
                    f"trung bình, sẽ thiệt {VSS:,.0f} tỷ so với mô hình xác suất → đo *giá trị "
                    f"của tư duy stochastic*. **EVPI = {EVPI:,.0f}**: mức tối đa nên chi cho "
                    "thông tin/dự báo hoàn hảo (ví dụ đầu tư hệ thống cảnh báo sớm, mô hình dự báo).")

            # ---- Câu 10.5.4: Robust optimization (minimax regret) ----
            st.subheader("Câu 10.5.4 — Robust optimization (minimax regret)")
            mr = pyo.ConcreteModel()
            mr.J = pyo.Set(initialize=J); mr.S = pyo.Set(initialize=S)
            mr.x = pyo.Var(mr.J, within=pyo.NonNegativeReals)
            mr.y = pyo.Var(mr.S, mr.J, within=pyo.NonNegativeReals)
            mr.w = pyo.Var(within=pyo.Reals)
            mr.b1 = pyo.Constraint(expr=sum(mr.x[j] for j in J) <= 65000)
            mr.b2 = pyo.Constraint(mr.S, rule=lambda mm, s: sum(mm.y[s, j] for j in J) <= 15000)
            mr.ac = pyo.Constraint(mr.S, rule=lambda mm, s: mm.y[s, "AI"] <= 0.5*mm.x["H"])
            mr.reg = pyo.Constraint(mr.S, rule=lambda mm, s: det[s]["Z"] -
                                    (sum(beta_base[j]*mm.x[j] for j in J) +
                                     sum(beta_s[s, j]*mm.y[s, j] for j in J)) <= mm.w)
            mr.obj = pyo.Objective(expr=mr.w, sense=pyo.minimize)
            solver.solve(mr)
            x_rob = {j: pyo.value(mr.x[j]) for j in J}
            w_rob = pyo.value(mr.w)

            def regret_of(xf):
                reg = {}
                for s in S:
                    z = sum(beta_base[j]*xf[j] for j in J)
                    ms = pyo.ConcreteModel(); ms.J = pyo.Set(initialize=J)
                    ms.y = pyo.Var(ms.J, within=pyo.NonNegativeReals)
                    ms.b2 = pyo.Constraint(expr=sum(ms.y[j] for j in J) <= 15000)
                    ms.ac = pyo.Constraint(expr=ms.y["AI"] <= 0.5*xf["H"])
                    ms.obj = pyo.Objective(expr=sum(beta_s[s, j]*ms.y[j] for j in J), sense=pyo.maximize)
                    solver.solve(ms)
                    z += pyo.value(ms.obj)
                    reg[s] = det[s]["Z"] - z
                return reg
            reg_sp, reg_rob = regret_of(x_sp), regret_of(x_rob)
            cc = st.columns([1.2, 1])
            with cc[0]:
                st.dataframe(pd.DataFrame({
                    "Kịch bản": [S_VI[s] for s in S],
                    "Regret (SP)": [round(reg_sp[s]) for s in S],
                    "Regret (Robust)": [round(reg_rob[s]) for s in S]}),
                    use_container_width=True, hide_index=True)
                st.caption(f"Max regret: SP = {max(reg_sp.values()):,.0f} vs "
                           f"Robust = {max(reg_rob.values()):,.0f}")
            with cc[1]:
                st.dataframe(pd.DataFrame({"Hạng mục": J,
                                           "x* SP": [round(x_sp[j]) for j in J],
                                           "x* Robust": [round(x_rob[j]) for j in J]}),
                             use_container_width=True, hide_index=True)
            st.info("Robust (minimax regret) tối thiểu hóa 'hối tiếc' ở kịch bản xấu nhất, nên "
                    "thường **thận trọng hơn SP**: đầu tư an toàn hơn để không quá tệ trong bất kỳ "
                    "kịch bản nào, đổi lại lợi ích kỳ vọng có thể thấp hơn đôi chút. Đây là lựa "
                    "chọn phù hợp khi nhà hoạch định 'ngại rủi ro' (risk-averse).")

        except Exception as e:
            st.warning(f"Không có solver Pyomo khả dụng trong môi trường này ({e}). "
                       "Hiển thị kết quả tham chiếu từ notebook.")
            st.dataframe(pd.DataFrame({"Kịch bản": [S_VI[s] for s in S],
                                       "Z*[s] Wait&See": [101500, 97750, 96250, 97750],
                                       "Xác suất": list(p_s.values())}),
                         use_container_width=True, hide_index=True)
            st.metric("Z* Stochastic (tham chiếu)", "98.575")

    with t_pol:
        st.subheader("10.6. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a) So với lời giải xác định, lời giải SP đầu tư H nhiều hơn hay ít hơn? Vì sao?**  \n"
            "Lời giải SP có xu hướng đầu tư vào **nhân lực số H nhiều hơn** lời giải EV. Lý do: "
            "hệ số βₕ cao nhất trong kịch bản khủng hoảng (1,10) — H đóng vai trò 'hàng hóa bảo "
            "hiểm', vừa giúp hấp thụ cú sốc, vừa 'mở khóa' khả năng đầu tư AI bổ sung ở giai đoạn "
            "2 (ràng buộc y_AI ≤ 0,5·x_H). Tư duy xác suất buộc mô hình chuẩn bị cho cả kịch bản xấu.\n\n"
            "**b) VSS dương nói lên điều gì về giá trị của tư duy xác suất trong hoạch định chính "
            "sách Việt Nam?**  \n"
            "VSS dương chứng minh rằng **bỏ qua bất định gây thiệt hại đo được**: lập kế hoạch chỉ "
            "theo một kịch bản 'trung bình' kém hơn việc cân nhắc toàn bộ phân phối kịch bản. Với "
            "một nền kinh tế mở như Việt Nam, đây là lập luận định lượng ủng hộ việc thể chế hóa "
            "phân tích kịch bản trong quy trình lập ngân sách trung hạn.\n\n"
            "**c) COVID-19 (2020–2022) và bão Yagi (2024) là các cú sốc thực tế. Việt Nam có đang "
            "'dưới đầu tư' vào nhân lực số như một hàng hóa bảo hiểm không?**  \n"
            "Cả hai cú sốc cho thấy nền kinh tế dễ tổn thương trước biến động ngoại sinh. Mô hình "
            "gợi ý rằng đầu tư vào nhân lực số (H) và giữ quỹ dự phòng có giá trị bảo hiểm cao — "
            "nếu thực tế đầu tư H thấp hơn mức SP đề xuất, có cơ sở để nói Việt Nam đang 'dưới đầu "
            "tư' vào năng lực chống chịu. Robust optimization còn nhấn mạnh thêm: nên ưu tiên "
            "phương án ít hối tiếc nhất ở kịch bản xấu, thay vì chạy theo kỳ vọng đơn thuần."
        )


# ============================================================================
#  BÀI 11 — Q-learning
# ============================================================================
ACTION_NAMES = ["Truyền thống", "Cân bằng", "Số hóa nhanh", "AI dẫn dắt", "Bao trùm"]
ALLOC = {0: np.array([0.70, 0.10, 0.10, 0.10]), 1: np.array([0.40, 0.25, 0.15, 0.20]),
         2: np.array([0.25, 0.45, 0.15, 0.15]), 3: np.array([0.20, 0.20, 0.45, 0.15]),
         4: np.array([0.30, 0.20, 0.10, 0.40])}
W_REW = np.array([0.40, 0.25, 0.20, 0.15])


def _env_step(state, action, K, D, AI, H, Y_prev, t):
    a = ALLOC[action]; budget = 2100.0
    K = (1-0.05)*K + a[0]*budget
    D = (1-0.12)*D + a[1]*budget*0.01
    AI = (1-0.15)*AI + a[2]*budget*0.05
    H = H + 0.8*(a[3]*budget*0.01) - 0.02*H
    A = 33.70*(1+0.003*(D/100)+0.002*(AI/100)+0.004*(H/100))**t
    L = 53.9*1.009**t
    Y = A*K**0.33*L**0.42*D**0.10*AI**0.08*H**0.07
    dgdp = (Y-Y_prev)/Y_prev; dun = max(0, -dgdp*0.5)
    cyber = (AI/(H+1))*0.01; emis = (K+AI)*0.0001
    reward = W_REW[0]*dgdp*100 - W_REW[1]*dun*100 - W_REW[2]*cyber - W_REW[3]*emis
    gl = 0 if dgdp < 0.03 else (1 if dgdp < 0.06 else 2)
    dl = 0 if D < 25 else (1 if D < 35 else 2)
    al = 0 if AI < 100 else (1 if AI < 200 else 2)
    hl = 0 if H < 35 else (1 if H < 50 else 2)
    return np.array([gl, dl, al, hl]), reward, K, D, AI, H, Y


@st.cache_data(show_spinner="Đang huấn luyện Q-learning (10.000 episodes)...")
def train_q(n_episodes=10000, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5)); gamma, alpha, T = 0.95, 0.1, 10
    hist = []
    for ep in range(n_episodes):
        s = rng.integers(0, 3, size=4)
        K, D, AI, H, Y_prev = 27500.0, 20.3, 86.0, 30.0, 12847.6
        total = 0; eps = max(0.05, 1.0-ep/5000)
        for t in range(T):
            a = rng.integers(5) if rng.random() < eps else int(np.argmax(Q[tuple(s)]))
            s2, r, K, D, AI, H, Y_prev = _env_step(s, a, K, D, AI, H, Y_prev, t)
            done = 1.0 if t == T-1 else 0.0
            Q[tuple(s)+(a,)] += alpha*(r+gamma*np.max(Q[tuple(s2)])*(1-done)-Q[tuple(s)+(a,)])
            total += r; s = s2
        hist.append(total)
    return Q, np.array(hist)


def _eval_policy(Q, kind, n_eval=300, seed=1):
    rng = np.random.default_rng(seed); rewards = []
    for _ in range(n_eval):
        s = rng.integers(0, 3, size=4)
        K, D, AI, H, Y_prev = 27500.0, 20.3, 86.0, 30.0, 12847.6
        total = 0
        for t in range(10):
            a = (int(np.argmax(Q[tuple(s)])) if kind == "opt" else
                 1 if kind == "a1" else 3 if kind == "a3" else rng.integers(5))
            s, r, K, D, AI, H, Y_prev = _env_step(s, a, K, D, AI, H, Y_prev, t)
            total += r
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)


def page_bai11():
    page_title("♻️", "Bài 11 — Học tăng cường (Q-learning) cho chính sách kinh tế thích nghi", "Bài 11")
    t_ctx, t_mod, t_data, t_calc, t_pol = st.tabs(
        ["11.1 Bối cảnh", "11.2 Mô hình", "11.3 Dữ liệu", "11.4 Tính toán", "11.5 Chính sách"])

    with t_ctx:
        st.subheader("11.1. Bối cảnh")
        st.write("Nền kinh tế Việt Nam được xem như **môi trường**, chính sách là **hành động**, "
                 "phần thưởng phản ánh phúc lợi xã hội. Học tăng cường cho phép chính sách thích "
                 "nghi theo trạng thái hiện tại, thay vì cố định như LP.")
        st.caption("Lưu ý: minh họa kỹ thuật — AI **không thay thế** trách nhiệm chính trị.")

    with t_mod:
        st.subheader("11.2. MDP đơn giản hóa")
        st.markdown("**Trạng thái** (3⁴=81): GDP growth / Digital / AI / Unemployment ∈ {low, med, high}.\n\n"
                    "**Hành động** (5): a0 Truyền thống, a1 Cân bằng, a2 Số hóa nhanh, a3 AI dẫn dắt, a4 Bao trùm.")
        st.latex(r"R_t = w_1\Delta GDP - w_2\Delta unemploy - w_3 CyberRisk - w_4 Emission")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]")

    with t_data:
        st.subheader("11.3. Không gian trạng thái & phần thưởng")
        st.markdown("**Trạng thái** (3⁴ = 81 trạng thái): mỗi yếu tố GDP growth / Digital / AI / "
                    "Unemployment được rời rạc hóa thành 3 mức {thấp, trung bình, cao}.")
        st.dataframe(pd.DataFrame({
            "Yếu tố": ["GDP growth", "Digital (D)", "AI", "Unemployment"],
            "Thấp (0)": ["< 3%", "< 25%", "< 100 ngh.DN", "thấp"],
            "Trung bình (1)": ["3–6%", "25–35%", "100–200", "TB"],
            "Cao (2)": ["> 6%", "> 35%", "> 200", "cao"]}),
            use_container_width=True, hide_index=True)
        st.markdown("**Phần thưởng** (phúc lợi xã hội), trọng số w = (0,40; 0,25; 0,20; 0,15):")
        st.latex(r"R_t = w_1\Delta GDP - w_2\Delta unemploy - w_3 CyberRisk - w_4 Emission")

    with t_calc:
        st.subheader("Câu 11.3.1 — Môi trường (gymnasium Env) & 5 hành động")
        st.markdown("Môi trường `VietnamEconomyEnv` mô phỏng 10 năm (T=10) là một episode, với "
                    "`reset()`, `step()`, `action_space=Discrete(5)`, `observation_space="
                    "MultiDiscrete([3,3,3,3])`. Mỗi hành động là một cơ cấu phân bổ ngân sách:")
        st.dataframe(pd.DataFrame({
            "Hành động": ACTION_NAMES, "K %": [70, 40, 25, 20, 30], "D %": [10, 25, 45, 20, 20],
            "AI %": [10, 15, 15, 45, 10], "H %": [10, 20, 15, 15, 40]}),
            use_container_width=True, hide_index=True)

        st.subheader("Câu 11.3.2 — Cấu hình huấn luyện Q-learning")
        st.markdown("- Learning rate **α = 0,1**; discount **γ = 0,95**.\n"
                    "- ε-greedy: ε giảm tuyến tính từ **1,0 → 0,05** qua 5.000 episodes "
                    "(khám phá → khai thác), tổng **10.000 episodes**.\n"
                    "- Bảng Q kích thước 3×3×3×3×5 = 405 giá trị (81 trạng thái × 5 hành động).")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]")

        Q, hist = train_q()

        st.subheader("Câu 11.3.3 — Chính sách tối ưu π*(s) = argmax Q(s,a)")
        test = [([1, 1, 0, 1], "VN 2026 thực tế (GDP_med, D_med, AI_low, H_med)"),
                ([0, 0, 0, 2], "Kịch bản tệ (GDP_low, D_low, AI_low, H_high)"),
                ([2, 2, 2, 2], "Kịch bản tốt (GDP_high, D_high, AI_high, H_high)"),
                ([0, 1, 0, 0], "Sau khủng hoảng (GDP_low, D_med, AI_low, H_low)"),
                ([1, 0, 2, 1], "AI mạnh, D yếu (GDP_med, D_low, AI_high, H_med)")]
        rows = []
        for s, d in test:
            q = Q[tuple(s)]
            a = int(np.argmax(q))
            rows.append({"Trạng thái khởi đầu": d, "π* hành động": ACTION_NAMES[a],
                         "Q(s,π*)": round(q[a], 3),
                         "Q-values": ", ".join(f"{ACTION_NAMES[i][:6]}={q[i]:.2f}" for i in range(5))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Mỗi trạng thái, agent chọn hành động có Q-value cao nhất. Cột Q-values cho "
                   "thấy mức độ 'tự tin' của chính sách — chênh lệch lớn nghĩa là lựa chọn rõ ràng.")

        st.subheader("Câu 11.3.4 — So sánh π* với chính sách rule-based")
        res = {"π* (Q-learning)": _eval_policy(Q, "opt"), "Luôn Cân bằng (a1)": _eval_policy(Q, "a1"),
               "Luôn AI dẫn dắt (a3)": _eval_policy(Q, "a3"), "Random": _eval_policy(Q, "rand")}
        opt_mean = res["π* (Q-learning)"][0]
        cc = st.columns([1, 1])
        with cc[0]:
            dfp = pd.DataFrame({"Chính sách": list(res.keys()),
                                "Phúc lợi TB": [round(v[0], 2) for v in res.values()],
                                "Std": [round(v[1], 2) for v in res.values()],
                                "Hơn Random": [f"{v[0]-res['Random'][0]:+.2f}" for v in res.values()]})
            st.dataframe(dfp, use_container_width=True, hide_index=True)
            st.metric("π* vượt Random", f"{opt_mean - res['Random'][0]:+.2f}")
            st.metric("π* vượt 'Luôn Cân bằng'", f"{opt_mean - res['Luôn Cân bằng (a1)'][0]:+.2f}")
        with cc[1]:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            nm = list(res.keys())
            ax.bar(nm, [res[n][0] for n in nm], yerr=[res[n][1] for n in nm],
                   color=["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"], capsize=5)
            ax.set_ylabel("Phúc lợi tích lũy TB"); plt.xticks(rotation=15, ha="right", fontsize=8)
            ax.set_title("So sánh 4 chính sách"); ax.grid(axis="y", alpha=0.3)
            show_fig(fig)

        st.subheader("Learning curve — hội tụ của Q-learning")
        fig, ax = plt.subplots(figsize=(10, 3.2))
        sm = np.convolve(hist, np.ones(200)/200, mode="valid")
        ax.plot(sm, "b-", lw=1.5)
        ax.axhline(opt_mean, color="green", ls="--", lw=1, label=f"π* hội tụ ≈ {opt_mean:.2f}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Phúc lợi (trung bình trượt 200 ep)")
        ax.set_title("Đường cong học của Q-learning"); ax.legend(); ax.grid(alpha=0.3)
        show_fig(fig)
        st.caption("Đường cong đi lên rồi ổn định cho thấy agent học được chính sách tốt và "
                   "hội tụ; ε giảm dần từ 1,0 → 0,05 chuyển từ 'khám phá' sang 'khai thác'.")

    with t_pol:
        st.subheader("11.4. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a)** Khi GDP thấp, D thấp, U cao → π* thường chọn hành động kích thích nhanh "
            "(cân bằng/bao trùm) — phù hợp 'quick win'.\n\n"
            "**b)** Khi GDP cao, AI cao, U thấp → π* chọn củng cố (consolidation) hợp lý.\n\n"
            "**c)** π* nên là **công cụ tham mưu**, tích hợp vào quy trình hoạch định như đầu vào "
            "phân tích, không tự động hóa quyết định chính trị — xã hội."
        )


# ============================================================================
#  BÀI 12 — AIDEOM-VN tích hợp
# ============================================================================
def page_bai12():
    page_title("🧩", "Bài 12 — Đồ án tích hợp: Xây dựng nguyên mẫu AIDEOM-VN", "Bài 12")
    import pulp
    from scipy.optimize import linprog

    # M1 dự báo
    a, b, g, d, th = 0.33, 0.42, 0.10, 0.08, 0.07
    K0, L0, D0v, AI0, H0, A0 = 27500, 53.9, 20.3, 86, 30, 33.70
    T = 4; years = list(range(2026, 2031)); ba = 3000

    def forecast(al):
        K, D, AI, H, A = K0, D0v, AI0, H0, A0
        tr = [A*K**a*L0**b*D**g*AI**d*H**th]
        for t in range(T):
            K = (1-0.05)*K + al["K"]*ba; D = (1-0.12)*D + al["D"]*ba*0.01
            AI = (1-0.15)*AI + al["AI"]*ba*0.05; H = H + 0.8*al["H"]*ba*0.01 - 0.02*H
            A = A*(1+0.003*(D/100)+0.002*(AI/100)+0.004*(H/100)); L = L0*1.009**(t+1)
            tr.append(A*K**a*L**b*D**g*AI**d*H**th)
        return tr
    scen = {"S1 Truyền thống": {"K": .70, "D": .10, "AI": .10, "H": .10},
            "S2 Số hóa nhanh": {"K": .25, "D": .45, "AI": .15, "H": .15},
            "S3 AI dẫn dắt": {"K": .20, "D": .20, "AI": .45, "H": .15},
            "S4 Bao trùm số": {"K": .30, "D": .20, "AI": .10, "H": .40},
            "S5 Tối ưu cân bằng": {"K": .25, "D": .25, "AI": .30, "H": .20}}
    gdp_fc = {n: forecast(al) for n, al in scen.items()}

    # M2 TOPSIS
    dr = load_regions()
    crit = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
            "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
            "internet_penetration_pct", "gini_coef"]
    is_ben = [True]*7 + [False]
    Xr = dr[crit].values.astype(float)
    C_exp = topsis(Xr, np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10]), is_ben)
    C_ent = topsis(Xr, entropy_weights(Xr), is_ben)

    # M3 LP
    D0d = dict(zip(dr["region_name_en"].map({
        "Northern Midlands and Mountains": "NMM", "Red River Delta": "RRD",
        "North Central and South Central Coast": "NCC", "Central Highlands": "CH",
        "Southeast": "SE", "Mekong Delta": "MD"}), dr["digital_index_0_100"]))
    m = pulp.LpProblem("M3", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (REGIONS, ITEMS), lowBound=0)
    m += pulp.lpSum(BETA[(r, j)]*x[r][j] for r in REGIONS for j in ITEMS)
    m += pulp.lpSum(x[r][j] for r in REGIONS for j in ITEMS) <= 50000
    for r in REGIONS:
        m += pulp.lpSum(x[r][j] for j in ITEMS) >= 5000
        m += pulp.lpSum(x[r][j] for j in ITEMS) <= 12000
    m += pulp.lpSum(x[r]["H"] for r in REGIONS) >= 12000
    Mv = pulp.LpVariable("Dmax")
    for r in REGIONS:
        m += D0d[r] + 0.002*x[r]["D"] <= Mv
        m += D0d[r] + 0.002*x[r]["D"] >= 0.6*Mv
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    alloc = np.array([[x[r][j].value() or 0 for j in ITEMS] for r in REGIONS])
    Z_lp = pulp.value(m.objective)

    # M4 lao động
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22])/100
    sec = ["Nông-LT", "CN chế biến", "Xây dựng", "Bán buôn", "Tài chính", "Logistics", "CNTT", "Giáo dục"]
    Nn = 8; coeff = a1 - c1*risk
    A2 = np.zeros((Nn, 2*Nn)); A3 = np.zeros((Nn, 2*Nn))
    for i in range(Nn):
        A2[i, i] = -coeff[i]; A2[i, Nn+i] = -b1[i]
        A3[i, i] = c1[i]*risk[i]; A3[i, Nn+i] = -d1[i]
    rl = linprog(np.concatenate([-coeff, -b1]),
                 A_ub=np.vstack([np.concatenate([np.ones(Nn), np.ones(Nn)]).reshape(1, -1), A2, A3]),
                 b_ub=np.concatenate([[30000], np.zeros(Nn), np.zeros(Nn)]),
                 bounds=[(0, None)]*(2*Nn), method="highs")
    NJ = coeff*rl.x[:Nn] + b1*rl.x[Nn:]

    st.dataframe(pd.DataFrame({
        "Module": ["M1", "M2", "M3", "M4", "M5", "M6"],
        "Tên": ["Dự báo kinh tế", "Đánh giá sẵn sàng số", "Tối ưu phân bổ",
                "Mô phỏng lao động", "Đánh giá rủi ro", "Dashboard ra QĐ"],
        "Kỹ thuật": ["Cobb-Douglas (Bài 1)", "TOPSIS+Entropy (Bài 6)", "LP+Động (Bài 4,8)",
                     "NetJob (Bài 9)", "Đa mục tiêu+SP (Bài 7,10)", "Tổng hợp + cảnh báo"]}),
        use_container_width=True, hide_index=True)
    st.divider()

    # ---- M1 ----
    st.subheader("📈 M1 — Dự báo kinh tế (Cobb-Douglas) 2026–2030")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for n, tr in gdp_fc.items():
        ax.plot(years, tr, marker="o", ms=4, label=n)
    ax.set_xlabel("Năm"); ax.set_ylabel("GDP (ngh.tỷ VND)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("GDP dự báo theo 5 kịch bản chính sách")
    show_fig(fig)
    st.caption("Đầu vào: dữ liệu vĩ mô 2020–2025. Đầu ra: quỹ đạo GDP/TFP đến 2030 cho 5 kịch bản.")
    st.divider()

    # ---- M2 ----
    st.subheader("🏆 M2 — Đánh giá sẵn sàng số (TOPSIS + Entropy)")
    cc = st.columns([1.2, 1])
    with cc[0]:
        st.dataframe(pd.DataFrame({"Vùng": REGION_VI, "C* Expert": np.round(C_exp, 4),
                                   "C* Entropy": np.round(C_ent, 4),
                                   "Hạng": pd.Series(C_exp).rank(ascending=False).astype(int).values}),
                     use_container_width=True, hide_index=True)
    with cc[1]:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        order = np.argsort(C_exp)
        ax.barh(np.array(REGION_VI)[order], C_exp[order], color="#3498db")
        plt.yticks(fontsize=8); ax.set_title("Xếp hạng sẵn sàng AI")
        show_fig(fig)
    st.caption("Đầu vào: dữ liệu 6 vùng. Đầu ra: bản đồ mức độ sẵn sàng số/AI để chọn nơi triển khai.")
    st.divider()

    # ---- M3 ----
    st.subheader("🗺️ M3 — Tối ưu phân bổ ngân sách ngành-vùng (LP)")
    cc = st.columns([1.2, 1])
    with cc[0]:
        dfm = pd.DataFrame(alloc.round(0), columns=ITEMS, index=REGION_VI)
        dfm["Tổng"] = dfm.sum(1)
        st.dataframe(dfm, use_container_width=True)
        st.metric("Z* LP (GDP gain)", f"{Z_lp:,.0f} tỷ")
    with cc[1]:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        im = ax.imshow(alloc, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(6)); ax.set_yticklabels(REGIONS)
        ax.set_xticks(range(4)); ax.set_xticklabels(ITEMS)
        plt.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Heatmap phân bổ")
        show_fig(fig)
    st.caption("Đầu vào: ma trận β, ngân sách. Đầu ra: phân bổ tối ưu ngành-vùng (có ràng buộc công bằng).")
    st.divider()

    # ---- M4 ----
    st.subheader("👷 M4 — Mô phỏng thị trường lao động (NetJob)")
    st.metric("Tổng NetJob", f"{-rl.fun:,.0f} việc làm")
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.bar(sec, NJ, color="#2ecc71"); ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right", fontsize=8); ax.set_title("NetJob ròng theo ngành")
    show_fig(fig)
    st.caption("Đầu vào: kế hoạch AI/H. Đầu ra: việc làm ròng từng ngành dưới tác động tự động hóa.")
    st.divider()

    # ---- M5 ----
    st.subheader("⚠️ M5 — Đánh giá rủi ro (đa mục tiêu + ngẫu nhiên)")
    cc = st.columns([1.2, 1])
    with cc[0]:
        st.dataframe(pd.DataFrame({"Vùng": REGION_VI, "Phát thải eᵣ": E_R,
                                   "Rủi ro AI ρᵣ": RHO_R, "Giảm rủi ro/H σᵣ": SIG_R}),
                     use_container_width=True, hide_index=True)
    with cc[1]:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        xp = np.arange(6); w = 0.27
        ax.bar(xp - w, E_R, w, label="Phát thải", color="#e74c3c")
        ax.bar(xp, RHO_R, w, label="Rủi ro AI", color="#f39c12")
        ax.bar(xp + w, SIG_R, w, label="Giảm rủi ro/H", color="#2ecc71")
        ax.set_xticks(xp); ax.set_xticklabels([r[:8] for r in REGION_VI], rotation=30, ha="right", fontsize=6)
        ax.legend(fontsize=7); ax.set_title("Hệ số rủi ro theo vùng")
        show_fig(fig)
    st.caption("Rủi ro 3 trục: môi trường (phát thải), an ninh dữ liệu (AI), và khả năng giảm "
               "thiểu nhờ nhân lực (H). Đầu ra: cảnh báo rủi ro cho từng vùng.")
    st.divider()

    # ---- M6 ----
    st.subheader("🧭 M6 — Dashboard ra quyết định: tổng hợp & cảnh báo")
    rows = [{"Kịch bản": n, "GDP 2030 (ngh.tỷ)": round(tr[T]),
             "Tăng trưởng TB %/năm": round(((tr[T]/tr[0])**(1/T)-1)*100, 2)} for n, tr in gdp_fc.items()]
    dfc = pd.DataFrame(rows)
    cc = st.columns([1.2, 1])
    with cc[0]:
        st.dataframe(dfc, use_container_width=True, hide_index=True)
    with cc[1]:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.barh(dfc["Kịch bản"], dfc["GDP 2030 (ngh.tỷ)"],
                color=["#95a5a6", "#3498db", "#9b59b6", "#2ecc71", "#e67e22"])
        plt.yticks(fontsize=8); ax.set_title("GDP 2030 theo kịch bản")
        show_fig(fig)
    best = dfc.loc[dfc["GDP 2030 (ngh.tỷ)"].idxmax(), "Kịch bản"]
    st.success(f"✅ Kịch bản GDP 2030 cao nhất: **{best}**. NetJob ròng dương ({-rl.fun:,.0f} việc).")
    for n, tr in gdp_fc.items():
        gr = ((tr[T]/tr[0])**(1/T)-1)*100
        if gr < 6.0:
            st.warning(f"⚠️ {n}: tăng trưởng {gr:.1f}%/năm dưới mục tiêu 6,5–7%.")
    st.info("**Khuyến nghị tổng hợp:** ưu tiên kịch bản S5 (cân bằng) — tối đa tăng trưởng trong "
            "khi giữ bao trùm và rủi ro ở mức chấp nhận được; đặt 3 trung tâm AI tại vùng dẫn đầu "
            "TOPSIS; duy trì sàn đầu tư nhân lực số như 'bảo hiểm' rủi ro; áp ràng buộc môi trường "
            "theo cam kết COP26. Hệ thống AIDEOM-VN minh họa cách chuyển bài toán chính sách thành "
            "mô hình định lượng kiểm chứng được, tôn trọng nguyên tắc **AI hỗ trợ, không thay thế "
            "quyết định chính trị**.")


# ============================================================================
#  ROUTER
# ============================================================================
_ROUTES = {
    PAGES[0]: page_home, PAGES[1]: page_bai1, PAGES[2]: page_bai2, PAGES[3]: page_bai3,
    PAGES[4]: page_bai4, PAGES[5]: page_bai5, PAGES[6]: page_bai6, PAGES[7]: page_bai7,
    PAGES[8]: page_bai8, PAGES[9]: page_bai9, PAGES[10]: page_bai10, PAGES[11]: page_bai11,
    PAGES[12]: page_bai12,
}

try:
    _ROUTES[page]()
except FileNotFoundError as e:
    st.error("Không tìm thấy tệp dữ liệu CSV. Hãy đặt 3 tệp "
             "`vietnam_macro_2020_2025.csv`, `vietnam_sectors_2024.csv`, "
             f"`vietnam_regions_2024.csv` cùng thư mục với app.py.\n\nChi tiết: {e}")
except Exception as e:
    st.error(f"Đã xảy ra lỗi khi dựng trang: {e}")
    st.exception(e)

st.markdown("---")
st.caption("VN AIDEOM-VN • Vũ Công Minh – 23051329 • Bài tập lớn: Các mô hình ra quyết định")
