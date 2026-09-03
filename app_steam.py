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
# Lasso 已剔除的類型/標籤仍留在對照表裡；若直接餵進選單，使用者選了會以為
# 有作用，實際上完全不影響預測（33 個 genre 只有 14 個、50 個 tag 只有 37 個
# 是模型特徵）。這裡先過濾成 selected_features 的子集。
_SEL = set(selected_features)
genre_map = {k: v for k, v in meta["genre_map"].items() if v in _SEL}
tag_map   = {k: v for k, v in meta["tag_map"].items() if v in _SEL}

# ── 先驗校正 ──────────────────────────────────────────────────────────
# 訓練時以 SMOTE 將三類重採樣為 1:1:1，模型輸出的機率因此帶著均勻先驗，
# 直接顯示會系統性高估爆款。推論時乘回真實市場先驗再正規化。
TRAIN_PRIOR  = np.asarray(meta.get("train_prior",  [1/3, 1/3, 1/3]), dtype=float)
MARKET_PRIOR = np.asarray(meta.get("market_prior", [0.707885, 0.19732, 0.094796]), dtype=float)
PRIOR_RATIO  = MARKET_PRIOR / TRAIN_PRIOR

def apply_prior(prob):
    """把 SMOTE 均勻先驗下的後驗機率校正回市場先驗。"""
    adj = np.asarray(prob, dtype=float) * PRIOR_RATIO
    total = adj.sum()
    return adj / total if total > 0 else np.asarray(prob, dtype=float)

TIERS = {
    0: ("🔴  Flop（滯銷）",        "< 20,000 owners",         "#E53935"),
    1: ("🟡  Normal（回本）",      "20,000 – 100,000 owners", "#F9A825"),
    2: ("🟢  Blockbuster（爆款）", "> 100,000 owners",        "#2E7D32"),
}

# 快速範例預設值
PRESETS = {
    "indie": dict(
        is_free=False, price=9.99, lang_count=5, release_month=3,
        is_multiplayer=False, achievements=15, dlc_count=0,
        screenshot_count=5, movie_count=1,
        genres=["Indie", "Adventure"],
        tags=["Singleplayer", "Indie", "Story Rich"],
    ),
    "aaa": dict(
        is_free=False, price=59.99, lang_count=20, release_month=11,
        is_multiplayer=True, achievements=50, dlc_count=3,
        screenshot_count=12, movie_count=2,
        genres=["Action", "Adventure"],
        tags=["Action", "Multiplayer", "Open World"],
    ),
    "f2p": dict(
        is_free=True, price=0.0, lang_count=12, release_month=6,
        is_multiplayer=True, achievements=30, dlc_count=0,
        screenshot_count=8, movie_count=2,
        genres=["Action", "Free To Play"],
        tags=["Action", "Multiplayer", "Shooter"],
    ),
}

# ── Session state 初始化 ──────────────────────────────────────────────
def apply_preset(key):
    for k, v in PRESETS[key].items():
        st.session_state[k] = list(v) if isinstance(v, list) else v

for k, v in dict(is_free=False, price=9.99, lang_count=5, release_month=9,
                 is_multiplayer=False, achievements=20, dlc_count=0,
                 screenshot_count=5, movie_count=1,
                 genres=["Indie","Action"],
                 tags=["Singleplayer","Action"]).items():
    if k not in st.session_state:
        st.session_state[k] = v

# 舊 session 可能存著已被過濾掉的選項，會讓 multiselect 的 default 落在
# options 之外而報錯，這裡清乾淨。
st.session_state["genres"] = [g for g in st.session_state["genres"] if g in genre_map]
st.session_state["tags"]   = [t for t in st.session_state["tags"]   if t in tag_map]

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

    ac1, ac2 = st.columns(2)
    with ac1:
        achievements = st.number_input(
            "成就數量", 0, 5000, key="achievements",
            help="重要性第 3 名 — 內容規模的代理指標",
        )
    with ac2:
        dlc_count = st.number_input(
            "DLC 數量", 0, 200, key="dlc_count",
            help="重要性第 5 名 — 對預測結果影響最大的單一輸入",
        )

    with st.expander("行銷素材（重要性較低）"):
        sc1, sc2 = st.columns(2)
        with sc1:
            screenshot_count = st.number_input("截圖數量", 0, 200, key="screenshot_count")
        with sc2:
            movie_count = st.number_input("影片數量", 0, 50, key="movie_count")

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
    row = {f: 0 for f in selected_features}
    row["price"]            = 0.0 if is_free else float(price)
    row["is_free"]          = int(is_free)
    row["lang_count"]       = int(lang_count)
    row["is_multiplayer"]   = int(is_multiplayer)
    row["release_month"]    = int(release_month)
    row["achievements"]     = int(achievements)
    row["dlc_count"]        = int(dlc_count)
    row["screenshot_count"] = int(screenshot_count)
    row["movie_count"]      = int(movie_count)

    for g in genres_selected:
        col = genre_map.get(g)
        if col and col in row: row[col] = 1

    for t in tags_selected[:10]:
        col = tag_map.get(t)
        if col and col in row: row[col] = 1

    X_in = pd.DataFrame([row])[selected_features].fillna(0).astype(np.float32)
    prob_raw = xgb_clf.predict_proba(X_in)[0]
    prob = apply_prior(prob_raw)
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
            ("🏆", "成就",   int(achievements)),
            ("📦", "DLC",    int(dlc_count)),
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
                Macro F1：0.5339 ｜ Accuracy：67.1%<br>
                機率已做先驗校正（訓練 SMOTE 1:1:1 → 市場 70.8 / 19.7 / 9.5）
            </div>
        </div>
        """, unsafe_allow_html=True)
