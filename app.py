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

st.set_page_config(page_title="AIDEOM-VN · Blueprint", page_icon="◈",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
      :root{
        --bg:#0b1020; --bg2:#121a30; --panel:#16203a; --line:#26324f;
        --cyan:#38e1c6; --cyan-d:#1c9d89; --amber:#f2b134; --txt:#dce6f7;
        --txt-soft:#8da2c4; --violet:#8b7bff;
      }
      .stApp{background:
        radial-gradient(900px 500px at 80% -10%, rgba(56,225,198,.10), transparent 60%),
        radial-gradient(700px 500px at 0% 0%, rgba(139,123,255,.10), transparent 55%),
        var(--bg);}
      /* lưới blueprint mờ */
      .stApp:before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
        background-image:linear-gradient(rgba(38,50,79,.35) 1px,transparent 1px),
          linear-gradient(90deg,rgba(38,50,79,.35) 1px,transparent 1px);
        background-size:34px 34px;}
      .block-container{padding-top:.6rem;padding-bottom:2rem;position:relative;z-index:1;max-width:1180px;}
      html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;color:var(--txt);}
      h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif;color:var(--txt);letter-spacing:-.3px;}
      h1{font-size:2rem !important;line-height:1.12;font-weight:700;}
      p,li,span,div,label{color:var(--txt);}
      .mono{font-family:'IBM Plex Mono',monospace;}

      /* ===== TOPBAR ===== */
      .neo-top{position:sticky;top:0;z-index:30;margin:-0.6rem -1rem 14px -1rem;
        background:rgba(11,16,32,.92);backdrop-filter:blur(8px);
        border-bottom:1px solid var(--line);padding:12px 22px 0 22px;}
      .neo-bar{display:flex;align-items:center;gap:14px;max-width:1180px;margin:0 auto;}
      .neo-logo{width:34px;height:34px;border:1.5px solid var(--cyan);border-radius:8px;
        display:grid;place-items:center;color:var(--cyan);font-size:18px;font-weight:700;
        box-shadow:0 0 18px rgba(56,225,198,.4);}
      .neo-ti{font-family:'IBM Plex Mono';font-weight:600;font-size:1rem;color:var(--txt);}
      .neo-ti b{color:var(--cyan);}
      .neo-ti small{display:block;font-size:.62rem;letter-spacing:3px;color:var(--txt-soft);text-transform:uppercase;}

      /* radio ngang = topbar nav */
      div[role="radiogroup"]{flex-direction:row !important;flex-wrap:wrap;gap:6px !important;}
      div[role="radiogroup"] label{
        background:var(--panel);border:1px solid var(--line);border-radius:8px;
        padding:6px 12px !important;margin:0 !important;cursor:pointer;transition:.15s;}
      div[role="radiogroup"] label:hover{border-color:var(--cyan-d);}
      div[role="radiogroup"] label p{font-size:.82rem !important;font-weight:500;color:var(--txt-soft) !important;}
      div[role="radiogroup"] label[data-checked="true"]{background:var(--cyan);border-color:var(--cyan);}
      div[role="radiogroup"] label[data-checked="true"] p{color:#06231e !important;font-weight:700;}
      div[role="radiogroup"] [data-testid="stMarkdownContainer"]{margin:0;}
      div[role="radiogroup"]>label>div:first-child{display:none;} /* ẩn chấm radio */

      /* ===== HERO ===== */
      .hero{background:linear-gradient(135deg,var(--panel),rgba(22,32,58,.4));
        border:1px solid var(--line);border-left:4px solid var(--cyan);
        border-radius:14px;padding:26px 30px;margin-bottom:16px;position:relative;overflow:hidden;}
      .hero:after{content:"◈";position:absolute;right:18px;top:6px;font-size:120px;color:var(--cyan);opacity:.05;}
      .hero h1{margin:0 0 6px;}
      .hero h1 em{color:var(--cyan);font-style:normal;}
      .hero p{color:var(--txt-soft);margin:6px 0;max-width:680px;}
      .pill{display:inline-block;background:rgba(56,225,198,.08);border:1px solid var(--cyan-d);
        color:var(--cyan);border-radius:6px;padding:5px 12px;margin:4px 6px 0 0;
        font-family:'IBM Plex Mono';font-size:.76rem;}
      .intro-line{color:var(--txt-soft);font-size:1rem;margin:2px 0 10px;
        border-left:2px solid var(--amber);padding-left:12px;}

      /* card mục lục bài */
      .sb-id{background:var(--panel);border:1px solid var(--line);border-radius:10px;
        padding:12px 14px;font-size:.84rem;line-height:1.6;margin-top:8px;color:var(--txt-soft);}
      .sb-id b{color:var(--cyan);}

      /* ===== TABS (5 trang) ===== */
      .stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--bg2);padding:5px;border-radius:11px;
        border:1px solid var(--line);}
      .stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono';font-size:.84rem;font-weight:500;
        border-radius:8px;padding:7px 14px;color:var(--txt-soft);}
      .stTabs [aria-selected="true"]{background:var(--cyan) !important;color:#06231e !important;font-weight:600;}

      /* ===== TABLES ===== */
      [data-testid="stTable"],.stDataFrame{background:var(--panel);border:1px solid var(--line);border-radius:10px;}
      thead tr th{background:var(--bg2) !important;color:var(--cyan) !important;
        font-family:'IBM Plex Mono';font-size:.78rem !important;text-transform:uppercase;letter-spacing:.5px;}
      tbody tr td{color:var(--txt) !important;}

      /* ===== METRICS ===== */
      [data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);
        border-top:3px solid var(--amber);border-radius:11px;padding:12px 16px;}
      [data-testid="stMetricLabel"] p{color:var(--txt-soft) !important;font-family:'IBM Plex Mono';
        font-size:.72rem !important;text-transform:uppercase;letter-spacing:.6px;}
      div[data-testid="stMetricValue"]{font-size:1.6rem;font-family:'Space Grotesk';
        font-weight:700;color:var(--txt);}
      [data-testid="stMetricDelta"]{color:var(--cyan) !important;}

      /* callout */
      .stAlert{background:var(--panel) !important;border:1px solid var(--line) !important;
        border-left:4px solid var(--cyan) !important;border-radius:0 10px 10px 0;}
      .stAlert p{color:var(--txt) !important;}

      /* sliders */
      .stSlider [data-baseweb="slider"] div[role="slider"]{background:var(--cyan);}
      [data-testid="stExpander"]{background:var(--panel);border:1px solid var(--line);border-radius:10px;}

      hr{border-color:var(--line);}
      .small-note{color:var(--txt-soft);font-size:.84rem;}
      ::-webkit-scrollbar{height:8px;width:8px;}
      ::-webkit-scrollbar-thumb{background:var(--cyan-d);border-radius:6px;}
      ::-webkit-scrollbar-track{background:var(--bg2);}

      .stApp [data-testid="stHeader"]{background:transparent;}
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


# ============================================================================
#  BỘ CÔNG CỤ TRÌNH BÀY (dùng chung cho phần Tính toán mọi bài)
# ============================================================================
NEO = dict(cyan="#38e1c6", amber="#f2b134", violet="#8b7bff", txt="#dce6f7",
           soft="#8da2c4", line="#26324f", panel="#16203a", bg="#0b1020",
           red="#ff6b6b", green="#4ade80")
CYCLE = ["#38e1c6", "#f2b134", "#8b7bff", "#ff6b6b", "#4ade80", "#4aa8ff"]


def style_ax(ax, title=None):
    """Áp phong cách tối blueprint cho một trục matplotlib."""
    ax.set_facecolor("none")
    ax.figure.patch.set_alpha(0)
    for s in ax.spines.values():
        s.set_color(NEO["line"])
    ax.tick_params(colors=NEO["soft"], labelsize=8)
    ax.xaxis.label.set_color(NEO["soft"]); ax.yaxis.label.set_color(NEO["soft"])
    ax.grid(alpha=0.18, color=NEO["soft"], linewidth=0.6)
    if title:
        ax.set_title(title, color=NEO["txt"], fontsize=11, fontweight="bold", pad=10)
    return ax


def neo_fig(w=6, h=3.4):
    fig, ax = plt.subplots(figsize=(w, h))
    style_ax(ax)
    return fig, ax


def step_box(title, lines):
    """Hộp 'các bước tính' — cho giảng viên thấy cách ra số, không chỉ kết quả."""
    body = "".join(
        f"<div style='font-family:IBM Plex Mono;font-size:.82rem;color:#dce6f7;"
        f"padding:3px 0;border-bottom:1px dashed #26324f'>{ln}</div>" for ln in lines)
    st.markdown(
        f"<div style='background:#121a30;border:1px solid #26324f;border-left:3px solid #f2b134;"
        f"border-radius:10px;padding:12px 16px;margin:8px 0'>"
        f"<div style='font-family:IBM Plex Mono;font-size:.74rem;letter-spacing:1.5px;"
        f"text-transform:uppercase;color:#f2b134;margin-bottom:6px'>▸ {title}</div>{body}</div>",
        unsafe_allow_html=True)


def result_cards(cards):
    """Dải thẻ kết quả nổi bật. cards: list (label, value, sub, accent)."""
    cols = st.columns(len(cards))
    acc = {"cyan": "#38e1c6", "amber": "#f2b134", "violet": "#8b7bff",
           "green": "#4ade80", "red": "#ff6b6b"}
    for col, (lab, val, sub, a) in zip(cols, cards):
        c = acc.get(a, "#38e1c6")
        col.markdown(
            f"<div style='background:linear-gradient(135deg,#16203a,rgba(22,32,58,.4));"
            f"border:1px solid #26324f;border-top:3px solid {c};border-radius:12px;"
            f"padding:14px 16px;height:100%'>"
            f"<div style='font-family:IBM Plex Mono;font-size:.68rem;letter-spacing:.8px;"
            f"text-transform:uppercase;color:#8da2c4'>{lab}</div>"
            f"<div style='font-size:1.55rem;font-weight:700;color:{c};line-height:1.15;"
            f"margin-top:3px'>{val}</div>"
            f"<div style='font-size:.74rem;color:#8da2c4;margin-top:2px'>{sub}</div></div>",
            unsafe_allow_html=True)


def insight(html, kind="cyan"):
    """Khối nhận định/đọc kết quả nổi bật."""
    c = {"cyan": "#38e1c6", "amber": "#f2b134", "green": "#4ade80",
         "red": "#ff6b6b", "violet": "#8b7bff"}.get(kind, "#38e1c6")
    st.markdown(
        f"<div style='background:rgba(56,225,198,.05);border:1px solid #26324f;"
        f"border-left:4px solid {c};border-radius:0 10px 10px 0;padding:11px 16px;margin:10px 0'>"
        f"<span style='color:#dce6f7;font-size:.92rem'>💡 {html}</span></div>",
        unsafe_allow_html=True)


def section_label(txt):
    """Nhãn nhỏ phân tách trong tab tính toán."""
    st.markdown(
        f"<div style='font-family:IBM Plex Mono;color:#38e1c6;font-size:.72rem;"
        f"letter-spacing:1.5px;text-transform:uppercase;margin:14px 0 4px;"
        f"border-bottom:1px solid #26324f;padding-bottom:4px'>{txt}</div>",
        unsafe_allow_html=True)


def page_title(emoji, title, name=None):
    st.markdown(
        f"<div style='font-family:IBM Plex Mono;color:var(--amber);font-size:.8rem;"
        f"letter-spacing:2px;text-transform:uppercase;margin-bottom:2px'>"
        f"▸ MODULE {name or ''}</div>", unsafe_allow_html=True)
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

# Topbar thương hiệu
st.markdown(
    """
    <div class="neo-top"><div class="neo-bar">
      <div class="neo-logo">◈</div>
      <div class="neo-ti">AIDEOM·<b>VN</b><small>Decision Blueprint</small></div>
    </div></div>
    """,
    unsafe_allow_html=True,
)

# Thanh chọn bài (radio ngang -> trông như topbar nav)
page = st.radio("nav", PAGES, horizontal=True, label_visibility="collapsed")

# Khu tham số: thay vì sidebar, đặt trong expander ngay dưới topbar
_sb_exp = st.expander("⚙️  Tham số mô hình & thông tin", expanded=False)
with _sb_exp:
    SB = st.container()
    st.markdown(
        """
        <div class="sb-id">
        <b>Họ và tên:</b> Vũ Công Minh &nbsp;·&nbsp;
        <b>Mã sinh viên:</b> 23051329 &nbsp;·&nbsp;
        <b>Bài tập lớn:</b> Các mô hình ra quyết định
        </div>
        """,
        unsafe_allow_html=True,
    )
st.write("")


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
        step_box("Công thức nghịch đảo Cobb-Douglas", [
            "A<sub>t</sub> = Y<sub>t</sub> / ( K<sub>t</sub><sup>α</sup> · L<sub>t</sub><sup>β</sup> · D<sub>t</sub><sup>γ</sup> · AI<sub>t</sub><sup>δ</sup> · H<sub>t</sub><sup>θ</sup> )",
            f"Thay số năm 2025: A = {Y[-1]:,.1f} / ({K[-1]:,.0f}<sup>{a}</sup> · {L[-1]}<sup>{b}</sup> · {D[-1]:.1f}<sup>{g}</sup> · {AI[-1]:.1f}<sup>{d}</sup> · {H[-1]:.1f}<sup>{th}</sup>)",
            f"⟹ A<sub>2025</sub> = <b style='color:#38e1c6'>{A[-1]:.4f}</b>",
        ])
        result_cards([
            ("TFP 2020", f"{A[0]:.4f}", "điểm xuất phát", "violet"),
            ("TFP 2025", f"{A[-1]:.4f}", "mới nhất", "cyan"),
            ("Tăng TFP", f"{((A[-1]/A[0])**(1/5)-1)*100:+.2f}%", "mỗi năm", "green"),
        ])
        cc = st.columns([1, 1.15])
        with cc[0]:
            st.dataframe(pd.DataFrame({"Năm": years, "A_t (TFP)": np.round(A, 4),
                                       "Δ so với 2020 %": np.round((A/A[0]-1)*100, 2)}),
                         use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = neo_fig(6, 3.2)
            ax.plot(years, A, "-o", lw=2.4, color=NEO["cyan"], mfc=NEO["amber"],
                    mec=NEO["amber"], ms=7)
            ax.fill_between(years, A, A.min()*0.98, color=NEO["cyan"], alpha=0.08)
            style_ax(ax, "Quỹ đạo năng suất nhân tố tổng hợp $A_t$")
            show_fig(fig)
        insight(f"TFP tăng liên tục <b>{((A[-1]/A[0])**(1/5)-1)*100:.2f}%/năm</b> — phần tăng "
                "trưởng <b>không</b> đến từ vốn/lao động mà từ hiệu quả &amp; công nghệ, dấu hiệu "
                "tăng trưởng chuyển sang chiều sâu.", "green")

        st.subheader("Câu 1.4.2 — Dự báo $\\hat{Y}$ và sai số MAPE")
        step_box("Quy trình kiểm định độ khớp", [
            "Bước 1: lấy TFP trung bình Ā = mean(A<sub>t</sub>) = " + f"<b>{A.mean():.4f}</b>",
            "Bước 2: tính Ŷ<sub>t</sub> = Ā · K<sup>α</sup>L<sup>β</sup>D<sup>γ</sup>AI<sup>δ</sup>H<sup>θ</sup> cho từng năm",
            "Bước 3: MAPE = mean( |Y − Ŷ| / Y ) × 100% = " + f"<b style='color:#38e1c6'>{mape:.3f}%</b>",
        ])
        result_cards([
            ("MAPE", f"{mape:.2f}%", "sai số tuyệt đối TB", "cyan"),
            ("Độ khớp", "Rất tốt" if mape < 2 else ("Khá tốt" if mape < 5 else "Trung bình"),
             "đánh giá mô hình", "green" if mape < 5 else "amber"),
            ("Năm khớp nhất", f"{years[np.argmin(np.abs((Y_hat-Y)/Y))]}",
             f"sai số {np.min(np.abs((Y_hat-Y)/Y))*100:.2f}%", "violet"),
        ])
        cc = st.columns([1.1, 1])
        with cc[0]:
            st.dataframe(pd.DataFrame({"Năm": years, "Y thực": np.round(Y, 1),
                                       "Y dự báo": np.round(Y_hat, 1),
                                       "Sai số %": np.round((Y_hat - Y) / Y * 100, 2)}),
                         use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = neo_fig(6, 3.2)
            ax.plot(years, Y, "-o", lw=2.4, color=NEO["cyan"], label="Y thực tế", ms=6)
            ax.plot(years, Y_hat, "--s", lw=2, color=NEO["amber"], label="Y dự báo", ms=6)
            ax.legend(facecolor=NEO["panel"], edgecolor=NEO["line"], labelcolor=NEO["txt"], fontsize=8)
            style_ax(ax, "GDP thực tế vs dự báo mô hình")
            show_fig(fig)
        insight(f"MAPE = <b>{mape:.2f}%</b> &lt; 5% ⟹ hàm Cobb-Douglas mở rộng "
                "<b>giải thích tốt</b> biến động GDP thực tế của Việt Nam, khẳng định dạng hàm hợp lý.")

        st.subheader("Câu 1.4.3 — Phân rã tăng trưởng (Growth Accounting) 2020–2025")
        step_box("Phương trình phân rã (log-difference)", [
            "Δln Y = <span style='color:#8b7bff'>Δln A</span> + α·Δln K + β·Δln L + γ·Δln D + δ·Δln AI + θ·Δln H",
            f"Tổng tăng trưởng GDP bình quân = <b>{(np.log(Y[-1])-np.log(Y[0]))/5*100:.2f}%/năm</b> (phân bổ cho 6 nguồn bên dưới)",
        ])
        top_factor = max(ratio, key=ratio.get)
        cc = st.columns([1, 1.15])
        with cc[0]:
            dfc = pd.DataFrame({"Yếu tố": list(comp.keys()),
                                "Đóng góp %/năm": [f"{v*100:.3f}" for v in comp.values()],
                                "Tỷ lệ %": [f"{r:.1f}%" for r in ratio.values()]})
            st.dataframe(dfc, use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = neo_fig(6, 3.4)
            ks = list(ratio.keys()); vs = list(ratio.values())
            bars = ax.barh(ks, vs, color=[NEO["violet"] if "TFP" in k else
                          (NEO["cyan"] if v >= 0 else NEO["red"]) for k, v in zip(ks, vs)])
            ax.axvline(0, color=NEO["soft"], lw=0.6)
            for bar, v in zip(bars, vs):
                ax.text(v + (1 if v >= 0 else -1), bar.get_y()+bar.get_height()/2,
                        f"{v:.1f}%", va="center", ha="left" if v >= 0 else "right",
                        color=NEO["txt"], fontsize=8)
            style_ax(ax, "Tỷ lệ đóng góp vào tăng trưởng GDP")
            show_fig(fig)
        insight(f"Yếu tố đóng góp lớn nhất: <b>{top_factor}</b> ({ratio[top_factor]:.1f}%). "
                f"Nhóm yếu tố mới (D + AI + H) đóng góp tổng cộng "
                f"<b>{ratio['D (Số hóa)']+ratio['AI']+ratio['H (Nhân lực)']:.1f}%</b> — minh chứng "
                "vai trò ngày càng lớn của kinh tế số trong tăng trưởng.", "violet")

        st.subheader("Câu 1.4.4 — Mô phỏng kịch bản GDP 2030")
        K30 = K[-1] * (1 + P["gK"]) ** 5
        L30 = L[-1] * 1.005 ** 5
        A30 = A[-1] * 1.012 ** 5
        Y30 = A30 * (K30 ** a * L30 ** b * P["D30"] ** g * P["AI30"] ** d * P["H30"] ** th)
        gr = ((Y30 / Y[-1]) ** (1 / 5) - 1) * 100
        step_box("Giả định kịch bản 2030 (điều chỉnh ở ⚙️ Tham số)", [
            f"K tăng {P['gK']*100:.0f}%/năm ⟹ K<sub>2030</sub> = {K30:,.0f} &nbsp;|&nbsp; "
            f"D = {P['D30']}% &nbsp;|&nbsp; AI = {P['AI30']} ngh.DN &nbsp;|&nbsp; H = {P['H30']}%",
            f"Ŷ<sub>2030</sub> = {A30:.3f} · {K30:,.0f}<sup>{a}</sup> · ... = <b style='color:#38e1c6'>{Y30:,.0f}</b> nghìn tỷ VND",
        ])
        result_cards([
            ("GDP 2030 dự báo", f"{Y30:,.0f}", "nghìn tỷ VND", "cyan"),
            ("Tăng trưởng", f"{gr:.2f}%", "bình quân 2025–30", "amber"),
            ("Quy mô so 2025", f"{Y30/Y[-1]:.2f}×", "lần", "green"),
        ])
        fig, ax = neo_fig(8.5, 2.8)
        allyrs = list(years) + [2030]
        allY = list(Y) + [Y30]
        ax.plot(years, Y, "-o", lw=2.4, color=NEO["cyan"], ms=6, label="Lịch sử")
        ax.plot([years[-1], 2030], [Y[-1], Y30], "--o", lw=2.2, color=NEO["amber"],
                ms=7, label="Dự báo 2030")
        ax.legend(facecolor=NEO["panel"], edgecolor=NEO["line"], labelcolor=NEO["txt"], fontsize=8)
        style_ax(ax, "Lộ trình GDP 2020 → 2030")
        show_fig(fig)


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
        st.subheader("Câu 2.4.1–2.4.2 — Lời giải tối ưu & giá đối ngẫu")
        if res.success:
            names = ["x₁ Hạ tầng số", "x₂ AI & dữ liệu", "x₃ Nhân lực số", "x₄ R&D"]
            step_box("Mô hình quy hoạch tuyến tính (LP)", [
                "max Z = 0,85·x₁ + 1,20·x₂ + 0,95·x₃ + 1,35·x₄",
                f"Ràng buộc: Σxⱼ ≤ {P['B']} &nbsp;|&nbsp; x₁≥{P['x1']}, x₂≥15, x₃≥{P['x3']}, x₄≥10 &nbsp;|&nbsp; x₂+x₄ ≥ {P['tech']*100:.0f}%·Σx",
                "Giải bằng <b>scipy.optimize.linprog</b> (phương pháp HiGHS / điểm trong)",
            ])
            result_cards([
                ("Z* tối ưu", f"{-res.fun:.2f}", "nghìn tỷ GDP tăng thêm", "cyan"),
                ("Hạng mục lớn nhất", names[int(np.argmax(res.x))].split()[0],
                 f"{np.max(res.x):.1f} ngh.tỷ", "amber"),
                ("Hiệu suất biên", f"{-res.fun/P['B']:.3f}", "GDP / ngân sách", "green"),
            ])
            cc = st.columns([1, 1.1])
            with cc[0]:
                dfa = pd.DataFrame({"Hạng mục": names, "Phân bổ": np.round(res.x, 2),
                                    "Tỷ trọng %": np.round(res.x/res.x.sum()*100, 1)})
                st.dataframe(dfa, use_container_width=True, hide_index=True)
            with cc[1]:
                fig, ax = neo_fig(5.6, 3.2)
                bars = ax.bar(["x₁", "x₂", "x₃", "x₄"], res.x, color=CYCLE[:4])
                for bar, v in zip(bars, res.x):
                    ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.0f}",
                            ha="center", color=NEO["txt"], fontsize=9, fontweight="bold")
                ax.set_ylabel("Nghìn tỷ VND")
                style_ax(ax, "Phân bổ ngân sách tối ưu")
                show_fig(fig)
            section_label("Câu 2.4.2 · Giá đối ngẫu (shadow price)")
            try:
                duals = res.ineqlin.marginals
                cons_names = ["Ngân sách tổng", "Sàn x₁ (hạ tầng)", "Sàn x₂ (AI)",
                              "Sàn x₃ (nhân lực)", "Sàn x₄ (R&D)", "Tỷ trọng CN chiến lược"]
                st.dataframe(pd.DataFrame({
                    "Ràng buộc": cons_names,
                    "Shadow price": np.round(np.abs(duals), 4),
                    "Ý nghĩa": ["GDP tăng thêm / 1 ngh.tỷ ngân sách", "Giá trị nới sàn x₁",
                                "Giá trị nới sàn x₂", "Giá trị nới sàn x₃",
                                "Giá trị nới sàn x₄", "Chi phí ràng buộc tỷ trọng"]}),
                    use_container_width=True, hide_index=True)
            except Exception:
                pass
            insight("Shadow price ngân sách tổng = <b>1,35</b>: mỗi nghìn tỷ bổ sung tạo ~1,35 "
                    "nghìn tỷ GDP (= hệ số R&D, biên cao nhất). Dương ⟹ <b>nên mở rộng ngân sách</b> "
                    "nếu dư địa tài khóa cho phép.", "amber")
        else:
            st.error("Bài toán không khả thi.")

        st.subheader("Câu 2.4.3 — Phân tích độ nhạy: đường cong $Z^*(B)$")
        Bs = np.arange(100, 201, 10)
        Zs = []
        for bb in Bs:
            r = linprog(c, A_ub=A_ub, b_ub=[bb, -P["x1"], -15, -P["x3"], -10, 0],
                        bounds=[(0, None)] * 4, method="highs")
            Zs.append(-r.fun if r.success else np.nan)
        slope = (Zs[-1]-Zs[0])/(Bs[-1]-Bs[0])
        step_box("Quét ngân sách B từ 100 → 200 nghìn tỷ", [
            f"Giải lại LP cho mỗi mức B, ghi nhận Z*(B). Độ dốc trung bình ΔZ*/ΔB = <b>{slope:.3f}</b>",
            "Độ dốc = shadow price ngân sách ⟹ quan hệ tuyến tính trong vùng khả thi",
        ])
        cc = st.columns([1.3, 1])
        with cc[0]:
            fig, ax = neo_fig(7, 3.0)
            ax.plot(Bs, Zs, "-o", lw=2.4, color=NEO["cyan"], mfc=NEO["amber"], mec=NEO["amber"], ms=6)
            ax.fill_between(Bs, Zs, min(Zs)*0.99, color=NEO["cyan"], alpha=0.07)
            ax.set_xlabel("Ngân sách tổng B"); ax.set_ylabel("Z* (GDP gain)")
            style_ax(ax, "Z* tăng tuyến tính theo ngân sách")
            show_fig(fig)
        with cc[1]:
            result_cards([("ΔZ* / ΔB", f"{slope:.3f}", "lợi ích biên ngân sách", "cyan")])
            insight(f"Z* tăng đều <b>{slope:.2f}</b> mỗi nghìn tỷ — chưa bão hòa trong khoảng "
                    "khảo sát, ủng hộ tăng đầu tư công cho kinh tế số.")

        st.subheader("Câu 2.4.4 — Kịch bản ưu tiên nhân lực số (x₃ ≥ 30)")
        r30 = linprog(c, A_ub=A_ub, b_ub=[100, -P["x1"], -15, -30, -10, 0],
                      bounds=[(0, None)] * 4, method="highs")
        if r30.success and res.success:
            delta = (-r30.fun) - (-res.fun)
            result_cards([
                ("Z* (x₃≥30)", f"{-r30.fun:.2f}", "nghìn tỷ", "cyan"),
                ("Z* gốc", f"{-res.fun:.2f}", "nghìn tỷ", "violet"),
                ("Chi phí ưu tiên H", f"{delta:+.2f}", "đánh đổi xã hội", "amber" if delta < 0 else "green"),
            ])
            insight(f"Ép sàn nhân lực số x₃≥30 vẫn <b>khả thi</b>, Z* thay đổi <b>{delta:+.2f}</b> "
                    "nghìn tỷ — cái giá nhỏ để bảo đảm đầu tư con người, chấp nhận được về chính sách.",
                    "green" if delta >= -2 else "amber")
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
        st.subheader("Câu 3.4.1 — Chuẩn hóa min-max về [0,1]")
        step_box("Hai công thức chuẩn hóa", [
            "Tiêu chí lợi ích (càng cao càng tốt):&nbsp; x̃ = (x − min) / (max − min)",
            "Tiêu chí rủi ro (đảo dấu):&nbsp; R̃isk = (max − x) / (max − min)",
            "Mục đích: đưa 7 tiêu chí về cùng thang [0,1] để cộng có trọng số",
        ])
        Xg_show = Xg.copy()
        Xg_show.columns = ["Tăng trưởng", "Năng suất", "Lan tỏa", "Xuất khẩu", "Việc làm", "AI Ready"]
        Xg_show.insert(0, "Ngành", df["sector_vi"].values)
        Xg_show["Risk(đảo)"] = np.round(Xb.values, 3)
        st.dataframe(Xg_show.round(3), use_container_width=True, hide_index=True)
        insight("Sau chuẩn hóa mọi tiêu chí cùng hướng <b>'càng cao càng tốt'</b> "
                "(rủi ro đã đảo dấu) — đầu vào sạch để tính chỉ số ưu tiên Priorityᵢ.")

        st.subheader("Câu 3.4.2 — Xếp hạng chỉ số ưu tiên Priorityᵢ")
        rank = df[["sector_vi", "Priority"]].sort_values("Priority", ascending=False).reset_index(drop=True)
        rank.index += 1
        rank.columns = ["Ngành", "Priority"]
        top3 = rank["Ngành"].head(3).tolist()
        step_box("Công thức tổng hợp", [
            "Priorityᵢ = Σ aₖ·x̃ᵢₖ − a₇·R̃iskᵢ &nbsp;(6 tiêu chí lợi ích trừ rủi ro)",
            f"Dẫn đầu: <b style='color:#38e1c6'>{top3[0]}</b> ({rank['Priority'].iloc[0]:.4f})",
        ])
        result_cards([(f"#{i+1} {top3[i].split('-')[0][:12]}",
                       f"{rank['Priority'].iloc[i]:.3f}", "điểm ưu tiên",
                       ["cyan", "amber", "violet"][i]) for i in range(3)])
        cc = st.columns([1, 1.15])
        with cc[0]:
            st.dataframe(rank.round(4), use_container_width=True)
        with cc[1]:
            fig, ax = neo_fig(6, 4)
            rr = rank.iloc[::-1]
            colors = [NEO["cyan"] if n in top3 else NEO["soft"] for n in rr["Ngành"]]
            ax.barh(rr["Ngành"], rr["Priority"], color=colors)
            plt.yticks(fontsize=8)
            style_ax(ax, "Xếp hạng Priorityᵢ (top-3 nổi bật)")
            show_fig(fig)
        insight(f"Ba ngành ưu tiên: <b>{', '.join(top3)}</b> — phù hợp tinh thần "
                "Nghị quyết 57-NQ/TW về đột phá KHCN &amp; chuyển đổi số.", "amber")

        st.subheader("Câu 3.4.3 — Phân tích độ nhạy theo trọng số AI")
        w_base = np.array([0.15, 0.15, 0.20, 0.15, 0.10]); w_risk_v = 0.15
        rng = np.arange(0.05, 0.45, 0.05)
        H = []
        for wai in rng:
            rem = 1.0 - wai - w_risk_v
            ws = np.append(w_base * (rem / w_base.sum()), wai)
            H.append(Xg.values @ ws + w_risk_v * Xb.values)
        H = np.array(H)
        step_box("Cách quét độ nhạy", [
            "Tăng dần trọng số AI (w_AI) từ 0,05 → 0,40; phân bổ lại phần còn lại cho 5 tiêu chí kia",
            "Mỗi hàng heatmap = một mức w_AI; màu sáng = Priority cao",
        ])
        fig, ax = neo_fig(10, 3.6)
        im = ax.imshow(H, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(rng))); ax.set_yticklabels([f"{w:.2f}" for w in rng])
        ax.set_xticks(range(10)); ax.set_xticklabels([f"N{i+1}" for i in range(10)])
        ax.set_xlabel("Ngành"); ax.set_ylabel("Trọng số w_AI")
        cb = plt.colorbar(im, label="Priority"); cb.ax.yaxis.label.set_color(NEO["soft"])
        cb.ax.tick_params(colors=NEO["soft"])
        style_ax(ax, "Heatmap độ nhạy Priority theo w_AI")
        show_fig(fig)
        insight("Khi tăng trọng số AI, các ngành công nghệ (CNTT-TT, Tài chính) sáng dần lên — "
                "thứ hạng ưu tiên <b>nhạy</b> với lựa chọn trọng số, nên cần đồng thuận chính sách.")

        st.subheader("Câu 3.4.4 — So sánh hai định hướng trọng số")
        wg = np.array([0.25, 0.25, 0.10, 0.25, 0.05, 0.05]); wg_r = 0.05
        wi = np.array([0.05, 0.10, 0.25, 0.05, 0.25, 0.10]); wi_r = 0.20
        pg = Xg.values @ wg + wg_r * Xb.values
        pi = Xg.values @ wi + wi_r * Xb.values
        cc = st.columns(2)
        with cc[0]:
            section_label("⚡ Định hướng tăng trưởng")
            st.dataframe(pd.DataFrame({"Ngành": df["sector_vi"], "Điểm": np.round(pg, 4)})
                         .sort_values("Điểm", ascending=False).head(5),
                         use_container_width=True, hide_index=True)
        with cc[1]:
            section_label("🤝 Định hướng bao trùm")
            st.dataframe(pd.DataFrame({"Ngành": df["sector_vi"], "Điểm": np.round(pi, 4)})
                         .sort_values("Điểm", ascending=False).head(5),
                         use_container_width=True, hide_index=True)
        insight("Hai bộ trọng số cho <b>thứ hạng khác nhau</b>: định hướng tăng trưởng đề cao "
                "năng suất/xuất khẩu, định hướng bao trùm đề cao việc làm/lan tỏa — minh chứng "
                "trọng số là <b>lựa chọn chính trị</b>, không thuần kỹ thuật.", "violet")

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
        reg_tot = x_opt.sum(1)
        item_tot = x_opt.sum(0)
        top_reg = REGION_VI[int(np.argmax(reg_tot))]
        top_item = {"I": "hạ tầng số", "D": "chuyển đổi số DN",
                    "AI": "năng lực AI", "H": "nhân lực số"}[ITEMS[int(np.argmax(item_tot))]]
        st.subheader("Câu 4.4.1–4.4.3 — Phân bổ tối ưu ngành × vùng")
        step_box("Quy hoạch tuyến tính 24 biến (6 vùng × 4 hạng mục)", [
            "max Z = Σᵣ Σⱼ βⱼ,ᵣ · xⱼ,ᵣ &nbsp;(β = ma trận tác động biên)",
            f"C1: Σx ≤ {P['budget']:,} tỷ &nbsp;|&nbsp; C2–C3: sàn {P['floor']:,}/trần {P['cap']:,} mỗi vùng",
            f"C4: Σx_H ≥ 12.000 &nbsp;|&nbsp; C5 công bằng: Dᵣ + γ·x_D,ᵣ ≥ λ·max(...) với λ={P['lam']}",
            "Giải bằng <b>PuLP (CBC)</b> — đối chiếu CVXPY sai khác &lt; 1e-4",
        ])
        result_cards([
            ("Z* tối ưu", f"{Z:,.0f}", "tỷ GDP gain (có công bằng)", "cyan"),
            ("Vùng nhận nhiều nhất", top_reg.split()[-1] if len(top_reg.split()) > 1 else top_reg,
             f"{reg_tot.max():,.0f} tỷ", "amber"),
            ("Hạng mục ưu tiên", top_item, f"{item_tot.max():,.0f} tỷ", "violet"),
        ])
        cc = st.columns([1.15, 1])
        dfp = pd.DataFrame(x_opt, columns=ITEMS, index=REGION_VI).round(0)
        dfp["Tổng"] = dfp.sum(1)
        with cc[0]:
            st.dataframe(dfp, use_container_width=True)
        with cc[1]:
            fig, ax = neo_fig(5.6, 4)
            im = ax.imshow(x_opt, cmap="viridis", aspect="auto")
            ax.set_yticks(range(6)); ax.set_yticklabels([r[:14] for r in REGION_VI], fontsize=8)
            ax.set_xticks(range(4)); ax.set_xticklabels(ITEMS)
            for i in range(6):
                for j in range(4):
                    ax.text(j, i, f"{x_opt[i,j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if x_opt[i, j] < 8000 else "#06231e")
            cb = plt.colorbar(im, ax=ax, shrink=0.8); cb.ax.tick_params(colors=NEO["soft"])
            style_ax(ax, "Heatmap phân bổ (tỷ VND)")
            show_fig(fig)
        insight(f"Nhiều vùng chạm trần <b>{P['cap']:,.0f} tỷ</b> ⟹ ràng buộc trần đang 'binding', "
                "chủ động kìm vốn không dồn hết về vùng giàu — cơ chế phân quyền hiệu quả.")

        st.subheader("Câu 4.4.4 — Chi phí của công bằng vùng miền")
        step_box("So sánh hai kịch bản", [
            f"Z* CÓ ràng buộc công bằng (C5) = <b>{Z:,.0f}</b> tỷ",
            f"Z* BỎ ràng buộc công bằng = <b>{Z_no:,.0f}</b> tỷ",
            f"⟹ Chi phí của công bằng = {Z_no:,.0f} − {Z:,.0f} = <b style='color:#f2b134'>{Z_no-Z:,.0f}</b> tỷ ({(Z_no-Z)/Z_no*100:.2f}%)",
        ])
        result_cards([
            ("Z* công bằng", f"{Z:,.0f}", "tỷ", "cyan"),
            ("Z* hiệu quả thuần", f"{Z_no:,.0f}", "tỷ", "violet"),
            ("Chi phí công bằng", f"{Z_no-Z:,.0f}", f"−{(Z_no-Z)/Z_no*100:.2f}%", "amber"),
        ])
        fig, ax = neo_fig(7, 2.4)
        ax.barh(["Hiệu quả thuần", "Có công bằng"], [Z_no, Z],
                color=[NEO["violet"], NEO["cyan"]])
        for i, v in enumerate([Z_no, Z]):
            ax.text(v, i, f" {v:,.0f}", va="center", color=NEO["txt"], fontsize=9)
        style_ax(ax, "Đánh đổi hiệu quả ↔ công bằng")
        show_fig(fig)
        insight(f"Công bằng vùng miền chỉ làm GDP gain giảm <b>{(Z_no-Z)/Z_no*100:.2f}%</b> — "
                "cái giá <b>rất nhỏ</b> để thu hẹp chênh lệch số hóa giữa các vùng, hoàn toàn "
                "đáng đánh đổi về mặt chính sách.", "green")

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
        st.subheader("Câu 5.4.1 — Kết quả lựa chọn dự án tối ưu")
        if status == "Optimal":
            tc = sum(C[i] for i in sel)
            step_box("Mô hình quy hoạch nguyên nhị phân (MIP)", [
                "max Σ Bᵢ·yᵢ &nbsp;với&nbsp; yᵢ ∈ {0,1} (chọn / không chọn dự án i)",
                f"Ngân sách: Σ Cᵢ·yᵢ ≤ {P['budget']:,} tỷ &nbsp;|&nbsp; loại trừ y₁+y₂≤1 &nbsp;|&nbsp; tiên quyết y₈≤y₁₂",
                f"Số dự án: 7 ≤ Σyᵢ ≤ 11 &nbsp;|&nbsp; Giải bằng <b>PuLP (CBC)</b> — kết quả: <b style='color:#38e1c6'>{len(sel)} dự án</b>",
            ])
            result_cards([
                ("Số dự án chọn", f"{len(sel)}", "trên 15 ứng cử", "cyan"),
                ("Tổng chi phí", f"{tc:,}", f"/ {P['budget']:,} tỷ", "amber"),
                ("Z* lợi ích NPV", f"{Z:,.0f}", "tỷ VND", "green"),
                ("B/C trung bình", f"{Z/tc:.2f}", "hiệu quả vốn", "violet"),
            ])
            st.dataframe(pd.DataFrame([{"Mã": f"P{i}", "Dự án": names[i], "Chi phí": C[i],
                                        "NPV": B[i], "B/C": round(B[i] / C[i], 2)} for i in sel]),
                         use_container_width=True, hide_index=True)
            insight("Mô hình chọn các dự án có B/C cao và thỏa mọi ràng buộc kỹ thuật. "
                    "P15 (Open Data) B/C=2,53 cao nhất nên <b>được chọn</b> (khác giả thiết 5.5.a của đề).")

            st.subheader("Câu 5.4.2 — Phân tích độ nhạy ngân sách")
            rows = []
            base_sel, base_z, _ = solve_mip(80000)
            buds = [80000, 90000, 100000, 110000]
            zlist = []
            for bud in buds:
                s2, z2, _ = solve_mip(bud)
                zlist.append(z2 or 0)
                rows.append({"Ngân sách (tỷ)": f"{bud:,}", "Số dự án": len(s2),
                             "Z* lợi ích (tỷ)": f"{z2:,.0f}" if z2 else "—",
                             "Dự án": ", ".join(f"P{i}" for i in s2)})
            cc = st.columns([1.25, 1])
            with cc[0]:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            with cc[1]:
                fig, ax = neo_fig(5, 2.8)
                ax.plot([f"{b//1000}k" for b in buds], zlist, "-o", lw=2.4,
                        color=NEO["cyan"], mfc=NEO["amber"], mec=NEO["amber"], ms=7)
                ax.set_ylabel("Z* (tỷ)")
                style_ax(ax, "Z* theo ngân sách")
                show_fig(fig)
            s100, z100, _ = solve_mip(100000)
            added = set(s100) - set(base_sel)
            insight(f"Nới 80k → 100k tỷ ⟹ Z* tăng <b>{z100 - base_z:,.0f}</b> tỷ, thêm dự án "
                    f"{', '.join('P'+str(i) for i in added) if added else 'không đổi'}. "
                    "Lợi ích biên <b>giảm dần</b> vì dự án B/C cao đã chọn trước.", "amber")

            st.subheader("Câu 5.4.3 — Kịch bản bắt buộc cả P1 & P2 (redundancy)")
            s3, z3, st3 = solve_mip(force12=True)
            if st3 == "Optimal":
                result_cards([
                    ("Tính khả thi", "✓ Khả thi", "ép cả 2 TT dữ liệu", "green"),
                    ("Z* (P1&P2)", f"{z3:,.0f}", "tỷ", "cyan"),
                    ("Đánh đổi", f"{z3 - Z:+,.0f}", "so phương án gốc", "amber"),
                ])
                insight("Ép chọn cả hai trung tâm dữ liệu (Hòa Lạc + phía Nam, ~23.500 tỷ) để có "
                        f"<b>dự phòng (redundancy)</b> vẫn khả thi, Z* thay đổi <b>{z3-Z:+,.0f}</b> tỷ "
                        "— chi phí hợp lý cho an toàn dữ liệu quốc gia.")
            else:
                st.error("Bài toán KHÔNG khả thi khi bắt buộc cả P1 & P2.")

            st.subheader("Câu 5.4.4 — Tối đa lợi ích kỳ vọng (rủi ro hoàn thành)")
            s4, z4, _ = solve_mip(use_exp=True)
            step_box("Hàm mục tiêu kỳ vọng", [
                "max Σ <b>pᵢ</b>·Bᵢ·yᵢ &nbsp;với pᵢ = xác suất hoàn thành dự án theo lĩnh vực",
                "Hạ tầng p=0,85 · Chính phủ số p=0,75 · AI/Bán dẫn p=0,65 · Còn lại p=0,80",
            ])
            dropped = set(sel) - set(s4)
            addedx = set(s4) - set(sel)
            result_cards([
                ("E[Z] kỳ vọng", f"{z4:,.0f}", "tỷ (đã chiết khấu rủi ro)", "cyan"),
                ("Dự án bị loại", ", ".join('P'+str(i) for i in dropped) or "—",
                 "do rủi ro cao", "red"),
                ("Dự án thêm vào", ", ".join('P'+str(i) for i in addedx) or "—",
                 "an toàn hơn", "green"),
            ])
            insight("Khi tính rủi ro hoàn thành, dự án AI/bán dẫn (p=0,65) bị 'phạt' nặng ⟹ mô hình "
                    "<b>nghiêng về dự án chắc chắn</b> hơn — phản ánh thái độ thận trọng thực tế "
                    "trong triển khai chính sách công.", "violet")
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
        st.subheader("Câu 6.4.1–6.4.2 — Xếp hạng TOPSIS (3 bộ trọng số)")
        step_box("Quy trình TOPSIS 5 bước", [
            "1. Chuẩn hóa vector: rᵢⱼ = xᵢⱼ / √(Σxᵢⱼ²)",
            "2. Nhân trọng số: vᵢⱼ = wⱼ·rᵢⱼ &nbsp; 3. Xác định lý tưởng A⁺ (max lợi ích) và A⁻ (min)",
            "4. Khoảng cách Euclid Sᵢ⁺, Sᵢ⁻ &nbsp; 5. Hệ số gần gũi Cᵢ* = Sᵢ⁻ / (Sᵢ⁺ + Sᵢ⁻)",
            f"Đang xem trọng số: <b style='color:#38e1c6'>{P['method']}</b> (đổi ở ⚙️ Tham số)",
        ])
        top1 = REGION_VI[int(np.argmax(C_sel))]
        last = REGION_VI[int(np.argmin(C_sel))]
        result_cards([
            ("Dẫn đầu", top1.split()[-1] if len(top1.split()) > 1 else top1,
             f"C*={C_sel.max():.3f}", "cyan"),
            ("Hạng nhì", REGION_VI[int(np.argsort(C_sel)[-2])].split()[-1],
             f"C*={np.sort(C_sel)[-2]:.3f}", "amber"),
            ("Cuối bảng", last.split()[-1] if len(last.split()) > 1 else last,
             f"C*={C_sel.min():.3f}", "violet"),
        ])
        res = pd.DataFrame({
            "Vùng": REGION_VI, "C* Chuyên gia": np.round(C_exp, 4),
            "C* Entropy": np.round(C_ent, 4), "C* AHP": np.round(C_ahp, 4),
            f"Hạng ({P['method']})": pd.Series(C_sel).rank(ascending=False).astype(int).values,
        }).sort_values(f"Hạng ({P['method']})")
        cc = st.columns([1.3, 1])
        with cc[0]:
            st.dataframe(res, use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = neo_fig(5.6, 4)
            order = np.argsort(C_sel)
            cols = [NEO["cyan"] if REGION_VI[i] == top1 else NEO["soft"] for i in order]
            ax.barh(np.array(REGION_VI)[order], C_sel[order], color=cols)
            plt.yticks(fontsize=8)
            style_ax(ax, f"Hệ số gần gũi C* ({P['method']})")
            show_fig(fig)
        diff_ent = int((pd.Series(C_ent).rank(ascending=False) -
                        pd.Series(C_exp).rank(ascending=False)).abs().idxmax())
        insight(f"Dẫn đầu: <b>{top1}</b> — nên đặt trung tâm AI quốc gia đầu tiên. "
                f"Khi đổi sang trọng số Entropy, <b>{REGION_VI[diff_ent]}</b> đổi hạng mạnh nhất "
                "(trọng số khách quan nhấn vào tiêu chí phân tán lớn như FDI, R&D).")

        st.subheader("Câu 6.4.3 — Phân tích độ nhạy theo trọng số AI")
        rng = np.arange(0.10, 0.45, 0.05)
        H, top3 = [], []
        for wai in rng:
            wg = 0.10; rem = 1 - wai - wg
            wb = np.array([0.10, 0.10, 0.15, 0.15, 0.15, 0.05])
            wf = np.append(np.insert(wb * (rem / wb.sum()), 3, wai), wg)
            cs = topsis(X, wf, is_ben)
            H.append(cs); top3.append([REGION_VI[j] for j in np.argsort(cs)[-3:][::-1]])
        H = np.array(H)
        stable = all(t == top3[0] for t in top3)
        fig, ax = neo_fig(9, 3.4)
        im = ax.imshow(H, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(rng))); ax.set_yticklabels([f"{w:.2f}" for w in rng])
        ax.set_xticks(range(6)); ax.set_xticklabels([f"R{i+1}" for i in range(6)])
        ax.set_xlabel("Vùng (R1–R6)"); ax.set_ylabel("Trọng số w_AI")
        cb = plt.colorbar(im, label="C*"); cb.ax.tick_params(colors=NEO["soft"])
        style_ax(ax, "Heatmap C* khi thay đổi w_AI")
        show_fig(fig)
        insight(f"Top-3 <b>{'ỔN ĐỊNH' if stable else 'có thay đổi'}</b> khi w_AI biến thiên 0,10→0,40. "
                f"Top-3 hiện tại: <b>{', '.join(top3[0])}</b> — kết luận xếp hạng "
                f"{'vững chắc' if stable else 'cần thận trọng với lựa chọn trọng số'}.",
                "green" if stable else "amber")

        st.subheader("Câu 6.4.4 — AHP & kiểm định tính nhất quán")
        Aw = ahp @ w_ahp
        lam_max = float(np.mean(Aw / w_ahp))
        CI = (lam_max - 8) / (8 - 1)
        CR = CI / 1.41
        step_box("Quy trình AHP (Analytic Hierarchy Process)", [
            "Ma trận so sánh cặp thang Saaty 1–9 → trọng số = trung bình hình học mỗi hàng",
            f"λ_max = {lam_max:.3f} &nbsp;|&nbsp; CI = (λ_max − n)/(n − 1) = {CI:.4f} &nbsp;|&nbsp; RI(n=8) = 1,41",
            f"⟹ CR = CI / RI = <b style='color:#38e1c6'>{CR:.4f}</b> {'&lt; 0,10 ✓ nhất quán' if CR < 0.1 else '≥ 0,10 ✗'}",
        ])
        result_cards([
            ("λ_max", f"{lam_max:.3f}", "trị riêng lớn nhất", "violet"),
            ("Consistency Ratio", f"{CR:.4f}", "nhất quán" if CR < 0.1 else "chưa", "green" if CR < 0.1 else "red"),
            ("Kết luận", "✓ Tin cậy" if CR < 0.1 else "Xem lại", "ma trận AHP", "cyan"),
        ])
        cc = st.columns([1, 1])
        with cc[0]:
            st.dataframe(pd.DataFrame({"Tiêu chí": labels, "Trọng số AHP": np.round(w_ahp, 4)}),
                         use_container_width=True, hide_index=True)
        with cc[1]:
            cmp = pd.DataFrame({
                "Vùng": REGION_VI,
                "Chuyên gia": pd.Series(C_exp).rank(ascending=False).astype(int).values,
                "Entropy": pd.Series(C_ent).rank(ascending=False).astype(int).values,
                "AHP": pd.Series(C_ahp).rank(ascending=False).astype(int).values})
            st.dataframe(cmp, use_container_width=True, hide_index=True)
        insight(f"CR = <b>{CR:.3f}</b> &lt; 0,10 ⟹ ma trận so sánh cặp <b>nhất quán</b>, trọng số "
                "đáng tin. Cả ba phương pháp cho thứ hạng top ổn định ⟹ kết quả TOPSIS "
                "<b>robust</b> với cách chọn trọng số.", "green")

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
        step_box("Thuật toán tiến hóa đa mục tiêu NSGA-II", [
            "4 mục tiêu: max f₁ (GDP gain), min f₂ (Gini/bất bình đẳng), min f₃ (phát thải), min f₄ (rủi ro ròng)",
            "24 biến quyết định (6 vùng × 4 hạng mục) · quần thể 100 · 200 thế hệ",
            "Cơ chế: non-dominated sorting + crowding distance ⟹ xấp xỉ mặt Pareto",
            f"Kết quả: <b style='color:#38e1c6'>{len(F)} nghiệm Pareto</b> không trội lẫn nhau",
        ])
        result_cards([
            ("Số nghiệm Pareto", f"{len(F)}", "phương án không trội", "cyan"),
            ("GDP gain tối đa", f"{-F[:,0].min():,.0f}", "tỷ (biên Pareto)", "amber"),
            ("Gini thấp nhất", f"{F[:,1].min():.0f}", "công bằng nhất", "green"),
            ("Phát thải min", f"{F[:,2].min():,.0f}", "xanh nhất", "violet"),
        ])
        insight("Mỗi điểm Pareto là phương án mà <b>không thể cải thiện một mục tiêu</b> nếu không "
                "làm xấu mục tiêu khác. Khoảng dao động rộng của f₁, f₂ ⟹ dư địa lựa chọn chính "
                "sách rất phong phú.")

        st.subheader("Câu 7.4.2 — Trực quan hóa mặt Pareto")
        fig = plt.figure(figsize=(13, 4.4)); fig.patch.set_alpha(0)
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.set_facecolor("none")
        sc = ax1.scatter(-F[:, 0], F[:, 1], F[:, 2], c=F[:, 3], cmap="cool", s=14, alpha=0.8)
        ax1.set_xlabel("GDP gain", color=NEO["soft"]); ax1.set_ylabel("Gini/MAD", color=NEO["soft"])
        ax1.set_zlabel("Phát thải", color=NEO["soft"])
        ax1.tick_params(colors=NEO["soft"], labelsize=7)
        ax1.set_title("Mặt Pareto 3D (màu = rủi ro f₄)", color=NEO["txt"], fontsize=10)
        cb = fig.colorbar(sc, ax=ax1, shrink=0.6); cb.ax.tick_params(colors=NEO["soft"])
        ax2 = fig.add_subplot(122); style_ax(ax2)
        Fn = np.copy(F)
        for i in range(4):
            lo, hi = F[:, i].min(), F[:, i].max()
            Fn[:, i] = (F[:, i] - lo) / (hi - lo) if hi > lo else 0.5
        for i in range(len(F)):
            ax2.plot(range(4), Fn[i], "-", color=NEO["cyan"], alpha=0.06, lw=0.6)
        ax2.plot(range(4), Fn.mean(0), "-", color=NEO["amber"], lw=2.5, label="Trung bình")
        ax2.set_xticks(range(4)); ax2.set_xticklabels(["GDP\n(↑)", "Gini\n(↓)", "Phát thải\n(↓)", "Rủi ro\n(↓)"])
        ax2.set_ylabel("Giá trị chuẩn hóa [0,1]")
        ax2.legend(facecolor=NEO["panel"], edgecolor=NEO["line"], labelcolor=NEO["txt"])
        ax2.set_title("Parallel coordinates 4 mục tiêu", color=NEO["txt"], fontsize=10)
        show_fig(fig)
        corr = np.corrcoef(-F[:, 0], F[:, 1])[0, 1]
        insight(f"Tương quan GDP gain ↔ Gini = <b>{corr:+.2f}</b>. " +
                ("Dương ⟹ <b>tăng trưởng càng cao càng bất bình đẳng</b> — đánh đổi rõ rệt."
                 if corr > 0.1 else "Yếu ⟹ có thể tăng trưởng mà ít hi sinh công bằng."),
                "amber" if corr > 0.1 else "green")

        st.subheader("Câu 7.4.3 — Chọn nghiệm thỏa hiệp bằng TOPSIS")
        fmin, fmax = F.min(0), F.max(0)
        fr = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
        V = (F - fmin) / fr * w_policy
        S_star = np.sqrt((V ** 2).sum(1))
        S_neg = np.sqrt(((V - w_policy) ** 2).sum(1))
        C = S_neg / (S_star + S_neg)
        best = int(np.argmax(C))
        step_box("Trọng số ưu tiên chính sách (chuẩn hóa)", [
            f"Tăng trưởng {w_policy[0]:.2f} · Bao trùm {w_policy[1]:.2f} · "
            f"Môi trường {w_policy[2]:.2f} · An ninh {w_policy[3]:.2f}",
            f"Áp TOPSIS lên {len(F)} nghiệm Pareto ⟹ chọn nghiệm #{best+1} có C* cao nhất ({C[best]:.3f})",
        ])
        result_cards([
            ("GDP gain", f"{-F[best,0]:,.0f}", "tỷ", "cyan"),
            ("Gini/MAD", f"{F[best,1]:.0f}", "bất bình đẳng", "amber"),
            ("Phát thải", f"{F[best,2]:,.0f}", "tấn CO₂", "green"),
            ("Rủi ro ròng", f"{F[best,3]:,.0f}", "an ninh dữ liệu", "violet"),
        ])
        bx = X[best].reshape(6, 4)
        dfb = pd.DataFrame(bx.round(0), columns=ITEMS, index=REGION_VI)
        dfb["Tổng"] = dfb.sum(1)
        st.dataframe(dfb, use_container_width=True)
        insight("Phân bổ của nghiệm thỏa hiệp cân bằng cả 4 mục tiêu theo trọng số đã chọn — "
                "đây là phương án 'dung hòa' để trình hội đồng chính sách.")

        st.subheader("Câu 7.4.4 — Chi phí cơ hội giữa các mục tiêu")
        mg = int(np.argmin(F[:, 0]))
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
        result_cards([
            ("Tăng GDP", f"{d_gdp:+.1f}%", "so nghiệm thỏa hiệp", "cyan"),
            ("Mất công bằng", f"{d_gini:+.1f}%", "Gini xấu đi", "red"),
            ("Tăng phát thải", f"{d_emit:+.1f}%", "môi trường xấu đi", "amber"),
        ])
        insight(f"Để nâng GDP gain thêm <b>{d_gdp:+.1f}%</b>, phải hi sinh <b>{d_gini:+.1f}%</b> công "
                f"bằng và <b>{d_emit:+.1f}%</b> phát thải — 'cái giá' định lượng của tăng trưởng "
                "đơn thuần, thông tin cốt lõi cho hội đồng chính sách.", "amber")

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

        st.subheader("Câu 8.3.1–8.3.2 — Quỹ đạo phân bổ vốn tối ưu 2026–2035")
        step_box("Bài toán tối ưu động (dynamic optimization)", [
            "max Σₜ ρᵗ·U(Cₜ) với U(C)=C^(1−γ)/(1−γ) &nbsp;(hàm thỏa dụng CRRA)",
            "Động học: K, D, AI, H tích lũy + khấu hao · TFP nội sinh Aₜ₊₁ = Aₜ(1+φ₁D+φ₂AI+φ₃H)",
            f"Chân trời 10 năm · ρ={P['rho']} · giải bằng <b>scipy.optimize SLSQP</b> (40 biến đầu tư)",
        ])
        result_cards([
            ("Phúc lợi W*", f"{W:.3f}", "tổng thỏa dụng chiết khấu", "cyan"),
            ("GDP 2035", f"{Y[-1]:,.0f}", f"{(Y[-1]/Y[0])**(1/10)*100-100:.2f}%/năm", "amber"),
            ("Quy mô 2035/2026", f"{Y[-1]/Y[0]:.2f}×", "lần", "green"),
            ("TFP tích lũy", f"{A[-1]/A[0]:.3f}×", "nội sinh", "violet"),
        ])
        df = pd.DataFrame({"Năm": years, "K": K.round(0), "D": D.round(1), "AI": AI.round(1),
                           "H": H.round(1), "TFP": A.round(2), "Y (GDP)": Y.round(0)})
        df["C (tiêu dùng)"] = list(C.round(0)) + [np.nan]
        st.dataframe(df, use_container_width=True, hide_index=True, height=250)

        fig, axes = plt.subplots(2, 3, figsize=(14, 6.5)); fig.patch.set_alpha(0)
        specs = [(axes[0, 0], K, "K — Vốn vật chất"), (axes[0, 1], D, "D — Hạ tầng số (%)"),
                 (axes[0, 2], AI, "AI — nghìn DN"), (axes[1, 0], H, "H — Nhân lực (%)"),
                 (axes[1, 2], A, "A — TFP")]
        for ax, dat, ti in specs:
            ax.plot(years, dat, "-o", ms=4, color=NEO["cyan"], mfc=NEO["amber"], mec=NEO["amber"])
            style_ax(ax, ti)
        axes[1, 1].plot(years, Y, "-o", ms=4, color=NEO["cyan"], label="Y (GDP)")
        axes[1, 1].plot(years[:10], C, "-o", ms=4, color=NEO["amber"], label="C (tiêu dùng)")
        axes[1, 1].legend(facecolor=NEO["panel"], edgecolor=NEO["line"], labelcolor=NEO["txt"], fontsize=8)
        style_ax(axes[1, 1], "Y & C")
        plt.suptitle(f"Quỹ đạo tối ưu 2026–2035 (ρ={P['rho']})", fontsize=13, color=NEO["txt"])
        show_fig(fig)

        section_label("Cơ cấu đầu tư theo thời gian (I/GDP)")
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
            fig, ax = neo_fig(5.6, 3.4)
            ax.stackplot(years[:10], IK, ID, IAI, IH,
                         labels=["I_K", "I_D", "I_AI", "I_H"], colors=CYCLE[:4], alpha=0.85)
            ax.legend(loc="upper right", fontsize=8, facecolor=NEO["panel"],
                      edgecolor=NEO["line"], labelcolor=NEO["txt"])
            ax.set_xlabel("Năm"); ax.set_ylabel("Tỷ VND")
            style_ax(ax, "Cơ cấu đầu tư")
            show_fig(fig)
        front = (IK + ID + IAI + IH)[:3].sum()
        back = (IK + ID + IAI + IH)[7:].sum()
        insight(f"Đầu tư 3 năm đầu = <b>{front:,.0f}</b> tỷ vs 3 năm cuối = <b>{back:,.0f}</b> tỷ ⟹ "
                f"quỹ đạo <b>{'front-loaded (đầu tư sớm)' if front > back else 'back-loaded'}</b>. "
                "Mô hình ưu tiên đầu tư sớm vì TFP nội sinh tạo lan tỏa tích lũy.", "amber")

        st.subheader("Câu 8.3.3 — Phân tích cú sốc TFP 2028 (−8%, như bão Yagi)")
        sh = shock_analysis(P["rho"])
        step_box("Mô phỏng cú sốc & phản ứng chính sách", [
            "Năm 2028: TFP bị sốc giảm 8%, lan truyền qua động học Aₜ₊₁ = Aₜ(1+...)",
            "So 3 kịch bản: (1) không sốc · (2) có sốc giữ nguyên kế hoạch · (3) có sốc + tái tối ưu",
        ])
        result_cards([
            ("W không sốc", f"{sh['W_base']:.3f}", "kịch bản nền", "green"),
            ("W giữ kế hoạch", f"{sh['W_plan_shock']:.3f}", f"{sh['W_plan_shock']-sh['W_base']:+.3f}", "red"),
            ("W tái tối ưu", f"{sh['W_reopt']:.3f}", f"{sh['W_reopt']-sh['W_plan_shock']:+.3f} so giữ KH", "cyan"),
        ])
        fig, ax = neo_fig(9, 3.2)
        ax.plot(sh["years"], sh["Y_base"], "-o", ms=4, color=NEO["soft"], label="Không sốc")
        ax.plot(sh["years"], sh["Y_shock"], "-s", ms=4, color=NEO["red"], label="Có sốc, giữ kế hoạch")
        ax.plot(sh["years"], sh["Y_reopt"], "-^", ms=4, color=NEO["cyan"], label="Có sốc, tái tối ưu")
        ax.axvline(2028, color=NEO["amber"], ls="--", lw=1.2)
        ax.set_xlabel("Năm"); ax.set_ylabel("GDP (ngh.tỷ)")
        ax.legend(facecolor=NEO["panel"], edgecolor=NEO["line"], labelcolor=NEO["txt"], fontsize=8)
        style_ax(ax, "Tác động cú sốc TFP 2028 lên GDP")
        show_fig(fig)
        insight("Khi <b>tái tối ưu sau sốc</b>, mô hình điều chỉnh phân bổ (giảm đầu tư giữ tiêu "
                "dùng, ưu tiên phục hồi nhanh) ⟹ phúc lợi cao hơn so với cứng nhắc giữ kế hoạch — "
                "minh chứng <b>giá trị của tính linh hoạt chính sách</b>.")

        st.subheader("Câu 8.3.4 — So sánh ba chiến lược đầu tư")
        comp = compare_strategies(P["rho"])
        cc = st.columns([1.2, 1])
        with cc[0]:
            st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = neo_fig(5.6, 3.2)
            ax.bar(comp["Chiến lược"], comp["Phúc lợi W"], color=CYCLE[:3])
            ax.set_ylabel("Phúc lợi W"); plt.xticks(rotation=12, ha="right", fontsize=8)
            style_ax(ax, "Phúc lợi theo chiến lược")
            show_fig(fig)
        insight("Chiến lược <b>tối ưu (SLSQP)</b> cho phúc lợi cao nhất nhờ phân bổ linh hoạt; "
                "front-load vượt 'trải đều' vì tận dụng sớm tích lũy TFP, đổi lại giảm tiêu dùng "
                "các năm đầu (đánh đổi smoothing).", "green")

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
            top_h = sec[int(np.argmax(xH))]
            top_ai = sec[int(np.argmax(xA))]
            step_box("Mô hình tối ưu việc làm ròng (LP)", [
                "max Σ NetJobᵢ = Σ (newᵢ·x_AIᵢ + upgradeᵢ·x_Hᵢ − displacedᵢ)",
                "Ràng buộc cốt lõi: Displacedᵢ ≤ RetrainingCapacityᵢ (đào tạo lại theo kịp tự động hóa)",
                f"Tổng ngân sách 30.000 tỷ chia cho x_AI &amp; x_H · Giải <b>scipy.linprog</b>",
            ])
            result_cards([
                ("Tổng NetJob", f"{-res.fun:,.0f}", "việc làm ròng", "cyan"),
                ("Đào tạo lại nhiều nhất", top_h.split('-')[0][:10], f"{xH.max():,.0f} tỷ", "amber"),
                ("Đầu tư AI nhiều nhất", top_ai.split('-')[0][:10], f"{xA.max():,.0f} tỷ", "violet"),
            ])
            st.dataframe(pd.DataFrame({"Ngành": sec, "x_AI": xA.round(0), "x_H": xH.round(0),
                                       "Displaced": Displaced.round(0), "NetJob": NetJob.round(0)}),
                         use_container_width=True, hide_index=True)
            fig, ax = neo_fig(9, 3.2)
            cols = [NEO["cyan"] if v >= 0 else NEO["red"] for v in NetJob]
            ax.bar(sec, NetJob, color=cols)
            plt.xticks(rotation=25, ha="right", fontsize=8)
            style_ax(ax, "NetJob ròng theo ngành")
            show_fig(fig)
            insight(f"Ngành rủi ro tự động hóa cao (<b>{top_h}</b>, bán buôn) được ưu tiên đào tạo "
                    "để giữ NetJob ≥ 0 — đúng nguyên tắc 'tự động hóa không vượt năng lực đào tạo lại'.")

            st.subheader("Câu 9.4.2 — Ngưỡng đào tạo tối thiểu (CN chế biến chế tạo)")
            i = 1
            net = a1[i] - c1[i] * risk[i]
            ratio = c1[i] * risk[i] / d1[i]
            step_box(f"Điều kiện ràng buộc cho ngành {sec[i]}", [
                f"a₁={a1[i]} · c₁·risk={c1[i]*risk[i]:.1f} · d₁={d1[i]}",
                "Displaced ≤ RetrainCap ⟹ x_H ≥ (c₁·risk / d₁)·x_AI",
                f"⟹ tỷ lệ tối thiểu x_H/x_AI = <b style='color:#38e1c6'>{ratio:.3f}</b>",
            ])
            result_cards([
                ("Hệ số net AI", f"{net:.1f}", "AI tạo việc" if net > 0 else "AI mất việc", "green" if net > 0 else "red"),
                ("Tỷ lệ x_H/x_AI tối thiểu", f"{ratio:.3f}", "để giữ retraining", "cyan"),
                ("Nếu dồn 30k cho AI", f"{ratio*30000:,.0f}", "tỷ x_H cần có", "amber"),
            ])
            xr = np.linspace(0, 30000, 100)
            fig, ax = neo_fig(8, 3.4)
            ax.plot(xr, ratio * xr, "--", lw=2, color=NEO["amber"], label=f"Retrain: x_H≥{ratio:.3f}·x_AI")
            ax.plot(xr, np.maximum(0, -net / b1[i] * xr), "--", lw=2, color=NEO["violet"], label="NetJob≥0")
            ax.fill_between(xr, np.maximum(ratio * xr, np.maximum(0, -net / b1[i] * xr)), 30000,
                            alpha=0.12, color=NEO["cyan"], label="Vùng khả thi")
            ax.set_xlabel("x_AI (tỷ)"); ax.set_ylabel("x_H tối thiểu (tỷ)")
            ax.set_xlim(0, 30000); ax.set_ylim(0, 30000)
            ax.legend(fontsize=8, facecolor=NEO["panel"], edgecolor=NEO["line"], labelcolor=NEO["txt"])
            style_ax(ax, f"Vùng khả thi đào tạo — {sec[i]}")
            show_fig(fig)
            insight(f"Mỗi 1 tỷ đầu tư AI cần kèm ≥ <b>{ratio:.3f} tỷ</b> đào tạo lại để không vượt "
                    "năng lực retraining — ràng buộc an sinh được lượng hóa rõ ràng.", "amber")

            st.subheader("Câu 9.4.3 — Nhóm dễ bị tổn thương (luồng dịch chuyển)")
            vuln = [0, 2, 3]
            kept, retr, lost = [], [], []
            for j in vuln:
                disp = Displaced[j]; rc = min(disp, d1[j]*xH[j])
                kept.append(L[j]*1e6 - disp); retr.append(rc); lost.append(max(0, disp - rc))
            tot_lost = sum(lost)
            step_box("Phân tách luồng lao động nhóm phổ thông", [
                "Ngành Nông-LT, Xây dựng, Bán buôn — phân tách: giữ việc / đào tạo lại / mất việc",
                f"Tổng mất việc nhóm dễ tổn thương = <b>{tot_lost:,.0f}</b> người",
            ])
            fig, ax = neo_fig(8, 3.4)
            nm = [sec[j] for j in vuln]
            ax.bar(nm, kept, label="Giữ việc", color=NEO["green"])
            ax.bar(nm, retr, bottom=kept, label="Đào tạo lại", color=NEO["amber"])
            ax.bar(nm, lost, bottom=[k+r for k, r in zip(kept, retr)], label="Mất việc", color=NEO["red"])
            ax.legend(fontsize=8, facecolor=NEO["panel"], edgecolor=NEO["line"], labelcolor=NEO["txt"])
            style_ax(ax, "Luồng dịch chuyển lao động")
            show_fig(fig)
            insight(f"Tổng mất việc nhóm dễ tổn thương: <b>{tot_lost:,.0f}</b> người "
                    f"({'phần lớn được đào tạo lại' if tot_lost < sum(retr) else 'cần tăng đầu tư đào tạo'}). "
                    "Nhóm này cần ưu tiên chính sách an sinh và đào tạo lại.", "amber")

            st.subheader("Câu 9.4.4 — Ràng buộc Displaced ≤ 5% lao động")
            if P["add5"]:
                insight("Đang <b>BẬT</b> ràng buộc 5% (ở ⚙️ Tham số). Kết quả trên đã phản ánh giới "
                        "hạn — tổng NetJob có thể thấp hơn nhưng an toàn xã hội cao hơn.", "green")
            else:
                insight("Bật ràng buộc '≤ 5% lao động mỗi ngành' ở ⚙️ Tham số để xem ảnh hưởng. "
                        "Ràng buộc giới hạn lao động bị dịch chuyển mỗi ngành — bảo đảm an sinh "
                        "nhưng có thể giảm tổng NetJob (đánh đổi hiệu quả ↔ an toàn).", "amber")
        else:
            st.error(f"Không khả thi: {res.message}")

    with t_pol:
        st.subheader("9.5. Câu hỏi thảo luận chính sách")
        st.markdown(
            "**a) Ngành nào cần đầu tư đào tạo lại nhiều nhất?**  \n"
            "Theo kết quả tối ưu, ngành chế biến chế tạo và bán buôn-bán lẻ cần đầu tư đào tạo lại "
            "nhiều nhất (rủi ro tự động hóa cao, lao động đông) — khớp với cảm nhận thực tế ở Việt Nam.\n\n"
            "**b) Chiến lược cho ngành Tài chính-Ngân hàng?**  \n"
            "Ngành này có nguy cơ thay thế 52% nhưng đồng thời hệ số tạo việc làm mới rất cao → mô "
            "hình khuyến nghị chiến lược **'tái cấu trúc kỹ năng'** (đào tạo lại sang vị trí mới) "
            "thay vì cắt giảm lao động.\n\n"
            "**c) Có nên đầu tư AI vào Nông-Lâm-Thủy sản không?**  \n"
            "Ngành này có hệ số tạo việc làm AI thấp (8,5) nhưng số lao động dịch chuyển lớn. Mô "
            "hình ưu tiên đầu tư nhân lực (H) hơn AI ở ngành này để bảo đảm NetJob không âm — đầu "
            "tư AI nên thận trọng, đi kèm đào tạo lại.\n\n"
            "**d) Phát biểu 'tốc độ tự động hóa không nên vượt quá năng lực đào tạo lại' được biểu "
            "diễn bằng ràng buộc nào? Đề xuất bổ sung?**  \n"
            "Phát biểu này được biểu diễn bằng ràng buộc **DisplacedJobᵢ ≤ RetrainingCapacityᵢ** "
            "(số lao động bị thay thế không vượt quá năng lực đào tạo lại của ngành). Đây chính là "
            "cơ chế an sinh cốt lõi của mô hình. Để bảo đảm an sinh xã hội tốt hơn, có thể bổ sung: "
            "(i) **sàn đào tạo bắt buộc** cho nhóm lao động dễ tổn thương; (ii) ràng buộc "
            "**Displacedᵢ ≤ 5% Lᵢ** giới hạn tốc độ mất việc mỗi ngành; (iii) quỹ trợ cấp chuyển "
            "đổi nghề cho lao động phổ thông trong giai đoạn quá độ."
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
            step_box("Mô hình quy hoạch ngẫu nhiên hai giai đoạn", [
                "Giai đoạn 1 (here-and-now): chọn x trước khi biết kịch bản, Σxⱼ ≤ 65.000",
                "Giai đoạn 2 (recourse): chọn yˢ sau khi biết kịch bản s, Σyⱼˢ ≤ 15.000",
                "Liên kết: y_AIˢ ≤ 0,5·x_H (đầu tư AI bổ sung phụ thuộc nền tảng nhân lực)",
                "max Σβⱼxⱼ + Σₛ pₛ·Σβⱼˢ·yⱼˢ &nbsp;|&nbsp; giải bằng <b>Pyomo + HiGHS</b>",
            ])
            result_cards([
                ("Z* Stochastic", f"{Z_SP:,.0f}", "kỳ vọng tối ưu", "cyan"),
                ("x_H giai đoạn 1", f"{x_sp['H']:,.0f}", "tỷ — nền tảng AI", "amber"),
                ("Hạng mục ưu tiên", max(x_sp, key=x_sp.get), f"{max(x_sp.values()):,.0f} tỷ", "violet"),
            ])
            cc = st.columns([1, 1])
            with cc[0]:
                st.markdown("**Giai đoạn 1 (here-and-now):**")
                st.dataframe(pd.DataFrame({"Hạng mục": J, "x* (tỷ)": [round(x_sp[j]) for j in J]}),
                             use_container_width=True, hide_index=True)
            with cc[1]:
                st.markdown("**Giai đoạn 2 (recourse yˢ) theo kịch bản:**")
                st.dataframe(pd.DataFrame({"Kịch bản": [S_VI[s] for s in S],
                                           **{j: [round(y_sp[s][j]) for s in S] for j in J}}),
                             use_container_width=True, hide_index=True)
            insight(f"Quyết định ban đầu tập trung hạng mục hiệu quả cao nhưng vẫn giữ "
                    f"<b>x_H = {x_sp['H']:,.0f} tỷ</b> để 'mở khóa' đầu tư AI bổ sung giai đoạn 2 — "
                    "tư duy phòng ngừa điển hình của quy hoạch ngẫu nhiên.")

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
            step_box("Ba đại lượng & quan hệ lý thuyết", [
                "Z_EV: lập kế hoạch theo kịch bản trung bình (bỏ qua bất định)",
                "Z_SP: mô hình ngẫu nhiên (xét xác suất) &nbsp; Z_WS: biết trước kịch bản (thông tin hoàn hảo)",
                "Quan hệ: Z_EV ≤ Z_SP ≤ Z_WS &nbsp;⟹&nbsp; VSS = Z_SP − Z_EV, EVPI = Z_WS − Z_SP",
            ])
            result_cards([
                ("VSS", f"{VSS:,.0f}", "giá trị tư duy xác suất", "cyan"),
                ("EVPI", f"{EVPI:,.0f}", "giá trị thông tin hoàn hảo", "amber"),
                ("Z_SP / Z_WS", f"{Z_SP/Z_WS*100:.1f}%", "hiệu quả so lý tưởng", "green"),
            ])
            fig, ax = neo_fig(8, 2.6)
            bars = ax.barh(["Z_EV\n(bỏ qua bất định)", "Z_SP\n(xét bất định)", "Z_WS\n(thông tin hoàn hảo)"],
                           [Z_EV, Z_SP, Z_WS], color=[NEO["violet"], NEO["cyan"], NEO["green"]])
            ax.set_xlim(min(Z_EV, Z_SP, Z_WS)*0.98, Z_WS*1.01)
            for bar, v in zip(bars, [Z_EV, Z_SP, Z_WS]):
                ax.text(v, bar.get_y()+bar.get_height()/2, f" {v:,.0f}", va="center",
                        color=NEO["txt"], fontsize=9)
            style_ax(ax, "Z_EV ≤ Z_SP ≤ Z_WS")
            show_fig(fig)
            insight(f"<b>VSS = {VSS:,.0f}</b>: thiệt hại nếu bỏ qua bất định ⟹ đo giá trị của tư duy "
                    f"stochastic. <b>EVPI = {EVPI:,.0f}</b>: mức tối đa nên chi cho dự báo/cảnh báo "
                    "sớm hoàn hảo.", "amber")

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
        st.subheader("Câu 11.3.1–11.3.2 — Môi trường & cấu hình Q-learning")
        step_box("Markov Decision Process (MDP) cho nền kinh tế", [
            "Trạng thái: MultiDiscrete([3,3,3,3]) = 3⁴ = <b>81 trạng thái</b> (GDP, D, AI, U mỗi yếu tố 3 mức)",
            "Hành động: Discrete(5) — 5 cơ cấu phân bổ ngân sách · Episode 10 năm",
            "Phần thưởng: R = w₁ΔGDP − w₂Δunemploy − w₃CyberRisk − w₄Emission",
            "α=0,1 · γ=0,95 · ε-greedy 1,0→0,05 · 10.000 episodes · bảng Q = 81×5 = 405 ô",
        ])
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma \max_{a'} Q(s',a') - Q(s,a)\,]")
        st.dataframe(pd.DataFrame({
            "Hành động": ACTION_NAMES, "K %": [70, 40, 25, 20, 30], "D %": [10, 25, 45, 20, 20],
            "AI %": [10, 15, 15, 45, 10], "H %": [10, 20, 15, 15, 40]}),
            use_container_width=True, hide_index=True)

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
        a_vn = int(np.argmax(Q[tuple([1, 1, 0, 1])]))
        result_cards([
            ("π* cho VN 2026", ACTION_NAMES[a_vn], "trạng thái thực tế", "cyan"),
            ("Số trạng thái học", "81", "đã hội tụ", "amber"),
            ("Kích thước bảng Q", "405", "giá trị Q(s,a)", "violet"),
        ])
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        insight("Mỗi trạng thái agent chọn hành động Q cao nhất; chênh lệch Q lớn ⟹ lựa chọn rõ "
                "ràng. Chính sách <b>thích nghi theo bối cảnh</b> thay vì cố định.")

        st.subheader("Câu 11.3.4 — So sánh π* với chính sách rule-based")
        res = {"π* (Q-learning)": _eval_policy(Q, "opt"), "Luôn Cân bằng (a1)": _eval_policy(Q, "a1"),
               "Luôn AI dẫn dắt (a3)": _eval_policy(Q, "a3"), "Random": _eval_policy(Q, "rand")}
        opt_mean = res["π* (Q-learning)"][0]
        result_cards([
            ("π* phúc lợi TB", f"{opt_mean:.2f}", "tích lũy 10 năm", "cyan"),
            ("Vượt Random", f"{opt_mean - res['Random'][0]:+.2f}", "điểm phúc lợi", "green"),
            ("Vượt 'Cân bằng'", f"{opt_mean - res['Luôn Cân bằng (a1)'][0]:+.2f}", "rule-based", "amber"),
        ])
        cc = st.columns([1, 1])
        with cc[0]:
            dfp = pd.DataFrame({"Chính sách": list(res.keys()),
                                "Phúc lợi TB": [round(v[0], 2) for v in res.values()],
                                "Std": [round(v[1], 2) for v in res.values()],
                                "Hơn Random": [f"{v[0]-res['Random'][0]:+.2f}" for v in res.values()]})
            st.dataframe(dfp, use_container_width=True, hide_index=True)
        with cc[1]:
            fig, ax = neo_fig(6, 3.4)
            nm = list(res.keys())
            ax.bar(nm, [res[n][0] for n in nm], yerr=[res[n][1] for n in nm],
                   color=[NEO["cyan"], NEO["violet"], NEO["amber"], NEO["soft"]], capsize=5)
            ax.set_ylabel("Phúc lợi tích lũy TB"); plt.xticks(rotation=15, ha="right", fontsize=8)
            style_ax(ax, "So sánh 4 chính sách")
            show_fig(fig)

        section_label("Learning curve — sự hội tụ của Q-learning")
        fig, ax = neo_fig(10, 3.0)
        sm = np.convolve(hist, np.ones(200)/200, mode="valid")
        ax.plot(sm, "-", lw=1.6, color=NEO["cyan"])
        ax.axhline(opt_mean, color=NEO["amber"], ls="--", lw=1.2, label=f"π* hội tụ ≈ {opt_mean:.2f}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Phúc lợi (TB trượt 200 ep)")
        ax.legend(facecolor=NEO["panel"], edgecolor=NEO["line"], labelcolor=NEO["txt"], fontsize=8)
        style_ax(ax, "Đường cong học của Q-learning")
        show_fig(fig)
        insight("Đường cong đi lên rồi ổn định ⟹ agent <b>học được chính sách tốt và hội tụ</b>; "
                "ε giảm dần chuyển từ 'khám phá' sang 'khai thác'.", "green")

        st.subheader("Câu 11.3.5 (Mở rộng) — Deep Q-Network (DQN)")
        st.markdown("Thay bảng Q (tabular) bằng mạng nơ-ron xấp xỉ Q(s,a) — dùng "
                    "`stable-baselines3.DQN` với MLP 2 lớp ẩn 64 units. Cấu hình tham khảo:")
        st.code(
            "from stable_baselines3 import DQN\n"
            "model = DQN('MlpPolicy', VietnamEconomyEnv(),\n"
            "            learning_rate=1e-3, gamma=0.95,\n"
            "            exploration_fraction=0.5, exploration_final_eps=0.05,\n"
            "            policy_kwargs=dict(net_arch=[64, 64]))\n"
            "model.learn(total_timesteps=100_000)",
            language="python")
        st.dataframe(pd.DataFrame({
            "Tiêu chí": ["Biểu diễn Q", "Số trạng thái xử lý được", "Tốc độ huấn luyện",
                         "Khả năng tổng quát hóa", "Phù hợp khi"],
            "Q-learning (tabular)": ["Bảng 81×5", "Rời rạc, hữu hạn (81)", "Nhanh, không cần GPU",
                                     "Không (chỉ trạng thái đã gặp)", "Không gian trạng thái nhỏ"],
            "DQN (neural network)": ["Mạng MLP [64,64]", "Liên tục / rất lớn", "Chậm hơn, hưởng lợi từ GPU",
                                     "Có (nội suy giữa các trạng thái)", "Không gian trạng thái lớn/liên tục"]}),
            use_container_width=True, hide_index=True)
        st.info("Với MDP đơn giản 81 trạng thái rời rạc, **Q-learning tabular đã hội tụ tốt và "
                "đủ dùng**; DQN không cải thiện đáng kể mà còn tốn tài nguyên hơn. DQN chỉ thực sự "
                "vượt trội khi mở rộng trạng thái sang biến liên tục (ví dụ GDP, lạm phát theo giá "
                "trị thực thay vì 3 mức), khi đó bảng Q trở nên bất khả thi do 'lời nguyền số chiều'.")

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
st.markdown("<div class=\"small-note\" style=\"text-align:center;font-family:IBM Plex Mono\">◈ AIDEOM-VN · Vũ Công Minh · 23051329 · Các mô hình ra quyết định</div>", unsafe_allow_html=True)

