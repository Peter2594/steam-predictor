"""
Steam 上市成功預測系統 — Streamlit Demo
MDS 2026 Group 5
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

# ── 載入模型 ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    clf = joblib.load("model_xgb_clf.pkl")
    with open("model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return clf, meta

xgb_clf, meta = load_model()
selected_features = meta["selected_features"]
genre_map = meta["genre_map"]
tag_map   = meta["tag_map"]

TIERS = {
    0: ("🔴  Flop（滯銷）",        "< 20,000 owners",         "#E53935"),
    1: ("🟡  Normal（回本）",      "20,000 – 100,000 owners", "#F9A825"),
    2: ("🟢  Blockbuster（爆款）", "> 100,000 owners",        "#2E7D32"),
}

# 快速範例預設值
PRESETS = {
    "indie": dict(
        is_free=False, price=9.99, lang_count=5, release_month=3,
        is_multiplayer=False,
        genres=["Indie", "Adventure"],
        tags=["Singleplayer", "Indie", "2D"],
    ),
    "aaa": dict(
        is_free=False, price=59.99, lang_count=20, release_month=11,
        is_multiplayer=True,
        genres=["Action", "Adventure"],
        tags=["Action", "Multiplayer", "Open World"],
    ),
    "f2p": dict(
        is_free=True, price=0.0, lang_count=12, release_month=6,
        is_multiplayer=True,
        genres=["Action", "Free To Play"],
        tags=["Action", "Multiplayer", "Shooter"],
    ),
}

# ── Session state 初始化 ──────────────────────────────────────────────
def apply_preset(key):
    p = PRESETS[key]
    st.session_state["is_free"]        = p["is_free"]
    st.session_state["price"]          = p["price"]
    st.session_state["lang_count"]     = p["lang_count"]
    st.session_state["release_month"]  = p["release_month"]
    st.session_state["is_multiplayer"] = p["is_multiplayer"]
    st.session_state["genres"]         = p["genres"]
    st.session_state["tags"]           = p["tags"]

for k, v in dict(is_free=False, price=9.99, lang_count=5, release_month=9,
                 is_multiplayer=False, genres=["Indie","Action"],
                 tags=["Singleplayer","Action"]).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 頁面設定 ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Steam 銷量預測", page_icon="🎮", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0f1117; }
    .card {
        background: #1e2130; border-radius: 14px;
        padding: 22px 24px; margin-bottom: 16px;
        border: 1px solid #2d3247;
    }
    .card-title {
        font-size: 11px; font-weight: 700; letter-spacing: 1.4px;
        text-transform: uppercase; color: #6b7db3; margin-bottom: 16px;
    }
    .result-card {
        border-radius: 14px; padding: 26px 32px;
        margin: 4px 0 20px 0; border: 1px solid;
    }
    .bar-wrap { margin: 10px 0; }
    .bar-meta { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
    .bar-track { background:#2d3247; border-radius:99px; height:10px; overflow:hidden; }
    .bar-fill  { height:100%; border-radius:99px; }
    .chip-row  { display:flex; gap:8px; flex-wrap:wrap; margin-top:4px; }
    .chip {
        background:#2d3247; border-radius:99px; padding:5px 13px;
        font-size:13px; color:#c5cde8; display:flex; align-items:center; gap:5px;
    }
    .chip-val { font-weight:700; color:#e8eaf6; }
    header[data-testid="stHeader"] { background: transparent; }
    div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:10px 0 24px 0;">
    <div style="font-size:28px;font-weight:800;color:#e8eaf6;letter-spacing:-0.5px;">
        🎮 Steam 遊戲上市成功預測
    </div>
    <div style="font-size:14px;color:#7b8db7;margin-top:6px;">
        輸入遊戲發行前資訊 → 預測市場銷量等級（Percentile-Based 3-Tier · XGBoost）
    </div>
</div>
""", unsafe_allow_html=True)

# ── 快速範例 ───────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">⚡ 快速範例</div>', unsafe_allow_html=True)
ex1, ex2, ex3, _ = st.columns([1, 1, 1, 3])
with ex1:
    if st.button("🎮  Indie 小品", use_container_width=True):
        apply_preset("indie")
        st.rerun()
with ex2:
    if st.button("⚔️  AAA 大作", use_container_width=True):
        apply_preset("aaa")
        st.rerun()
with ex3:
    if st.button("🆓  F2P 射擊", use_container_width=True):
        apply_preset("f2p")
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ── 輸入區 ─────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-title">📦 基本資訊</div>', unsafe_allow_html=True)

    is_free = st.toggle("免費遊戲（F2P）", key="is_free")

    pc1, pc2 = st.columns(2)
    with pc1:
        price = st.number_input(
            "定價（USD）", min_value=0.0, max_value=999.0,
            step=1.0, format="%.2f", key="price", disabled=is_free,
        )
    with pc2:
        lang_count = st.number_input("支援語言數", 1, 100, key="lang_count")

    mc1, mc2 = st.columns(2)
    with mc1:
        release_month = st.selectbox(
            "預計發行月份", list(range(1, 13)), key="release_month",
            format_func=lambda m: f"{m} 月",
        )
    with mc2:
        is_multiplayer = st.toggle("含多人 / Co-op", key="is_multiplayer")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><div class="card-title">🏷️ 類型與標籤</div>', unsafe_allow_html=True)

    genres_selected = st.multiselect(
        "遊戲類型（Genre）", list(genre_map.keys()), key="genres",
    )
    tags_selected = st.multiselect(
        "遊戲標籤（Tags）", list(tag_map.keys()), key="tags",
        help="最多 10 個標籤會被納入模型",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── 預測按鈕 ───────────────────────────────────────────────────────────
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
predict = st.button("🚀  預測市場表現", use_container_width=True, type="primary")

# ── 預測結果 ───────────────────────────────────────────────────────────
if predict:
    # 固定不顯示的特徵使用合理預設值
    row = {f: 0 for f in selected_features}
    row["price"]            = 0.0 if is_free else float(price)
    row["is_free"]          = int(is_free)
    row["lang_count"]       = int(lang_count)
    row["is_multiplayer"]   = int(is_multiplayer)
    row["release_month"]    = int(release_month)
    row["achievements"]     = 20    # 固定預設
    row["dlc_count"]        = 0
    row["screenshot_count"] = 5
    row["movie_count"]      = 1

    for g in genres_selected:
        col = genre_map.get(g)
        if col and col in row: row[col] = 1

    for t in tags_selected[:10]:
        col = tag_map.get(t)
        if col and col in row: row[col] = 1

    X_in = pd.DataFrame([row])[selected_features].fillna(0).astype(np.float32)
    prob = xgb_clf.predict_proba(X_in)[0]
    pred = int(np.argmax(prob))

    tier_label, owners_range, color = TIERS[pred]

    res_left, res_right = st.columns([3, 2], gap="large")

    with res_left:
        st.markdown(f"""
        <div class="result-card"
             style="background:{color}18;border-color:{color}55;">
            <div style="font-size:11px;font-weight:700;letter-spacing:1.2px;
                        text-transform:uppercase;color:{color};opacity:.8;margin-bottom:6px;">
                預測結果
            </div>
            <div style="font-size:36px;font-weight:900;color:{color};line-height:1.15;">
                {tier_label}
            </div>
            <div style="font-size:14px;color:{color};opacity:.65;margin-top:8px;">
                預估銷售規模：{owners_range}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">信心分數</div>', unsafe_allow_html=True)
        for i, (lbl, _, clr) in TIERS.items():
            pct = float(prob[i])
            bold  = "700" if i == pred else "400"
            tclr  = "#e8eaf6" if i == pred else "#9aa3c0"
            st.markdown(f"""
            <div class="bar-wrap">
                <div class="bar-meta">
                    <span style="font-size:14px;font-weight:{bold};color:{tclr};">{lbl}</span>
                    <span style="font-size:14px;font-weight:700;color:{clr};">{pct*100:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="background:{clr};width:{pct*100:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with res_right:
        st.markdown('<div class="card"><div class="card-title">本次輸入摘要</div>', unsafe_allow_html=True)
        summary = [
            ("💰", "定價",   "Free" if is_free else f"${price:.2f}"),
            ("🌐", "語言數", int(lang_count)),
            ("👥", "多人",   "是" if is_multiplayer else "否"),
            ("📅", "月份",   f"{release_month} 月"),
        ]
        chips = "".join(
            f'<div class="chip">{ic} {nm} <span class="chip-val">{vl}</span></div>'
            for ic, nm, vl in summary
        )
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

        if genres_selected:
            st.markdown(f"<div style='margin-top:12px;font-size:12px;color:#7b8db7;'>類型：{', '.join(genres_selected)}</div>", unsafe_allow_html=True)
        if tags_selected:
            shown = ', '.join(tags_selected[:5]) + ('...' if len(tags_selected) > 5 else '')
            st.markdown(f"<div style='margin-top:4px;font-size:12px;color:#7b8db7;'>標籤：{shown}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">模型資訊</div>
            <div style="font-size:13px;color:#9aa3c0;line-height:2;">
                模型：XGBoost（3-Tier Classification）<br>
                特徵：Lasso 篩選後 60 維<br>
                標籤：Percentile-Based Model B<br>
                Macro F1：0.5339 ｜ Accuracy：67.1%
            </div>
        </div>
        """, unsafe_allow_html=True)
