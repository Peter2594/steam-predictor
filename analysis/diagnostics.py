"""
四項診斷檢查，一次跑完。

    python analysis/diagnostics.py --data modeling_table.csv

需要一份「已建好的建模表」CSV，欄位包含：
  - model_meta.json 的 selected_features（60 欄）
  - tier          標籤，0=Flop 1=Normal 2=Blockbuster
  - release_year  發行年份（檢查 3、4 需要）

檢查項目：
  1. 先驗校正對 Accuracy / Macro F1 的影響
  2. dlc_count 等事後變數是否灌水（消融對照）
  3. 標籤覆蓋率是否形成「新舊遊戲」捷徑
  4. 時序切分 vs 隨機切分

結果同時輸出到終端機與 analysis/out/diagnostics.md。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
TIER_NAMES = ["Flop", "Normal", "Blockbuster"]

# Windows 主控台預設 cp950，印到中文或 ≤ 會直接拋 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# 與 model_xgb_clf.pkl 相同的超參數
XGB_PARAMS = dict(
    n_estimators=300, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softprob", num_class=3,
    tree_method="hist", random_state=42, n_jobs=-1,
)

# 快照時點才知道的欄位 —— 上市前拿不到，疑似洩漏
POST_HOC = ["dlc_count", "achievements", "screenshot_count", "movie_count"]

_report_lines = []


def say(line=""):
    print(line)
    _report_lines.append(line)


def resample(X, y, seed=42):
    """訓練集平衡。有 imbalanced-learn 就用 SMOTE，否則退回類別權重。"""
    try:
        from imblearn.over_sampling import SMOTE
        Xb, yb = SMOTE(k_neighbors=5, random_state=seed).fit_resample(X, y)
        return Xb, yb, None
    except ImportError:
        counts = np.bincount(y, minlength=3)
        w = counts.sum() / (3 * np.maximum(counts, 1))
        return X, y, np.asarray([w[c] for c in y])


def fit_predict(X_tr, y_tr, X_te, seed=42):
    Xb, yb, weights = resample(X_tr, y_tr, seed)
    clf = XGBClassifier(**{**XGB_PARAMS, "random_state": seed})
    clf.fit(Xb, yb, sample_weight=weights)
    return clf, clf.predict_proba(X_te)


def score(y_true, proba, prior_ratio=None):
    if prior_ratio is not None:
        adj = proba * prior_ratio
        proba = adj / adj.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    return accuracy_score(y_true, pred), f1_score(y_true, pred, average="macro"), pred


def prior_ratio_from(y_train_original):
    """SMOTE 後訓練先驗為均勻；市場先驗取自訓練集原始分布。"""
    market = np.bincount(y_train_original, minlength=3) / len(y_train_original)
    return market / np.full(3, 1 / 3)


# -- 檢查 1 ------------------------------------------------------------
def check_prior_correction(X, y, features):
    say("## 檢查 1 — 先驗校正對指標的影響")
    say()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X[features], y, test_size=0.2, stratify=y, random_state=42)
    _, proba = fit_predict(X_tr, y_tr, X_te)
    ratio = prior_ratio_from(y_tr)

    say("| 版本 | Accuracy | Macro F1 |")
    say("|---|---|---|")
    results = {}
    for label, r in [("未校正（argmax 原始機率）", None), ("先驗校正後", ratio)]:
        acc, mf1, pred = score(y_te, proba, r)
        results[label] = (acc, mf1, pred)
        say("| %s | %.4f | %.4f |" % (label, acc, mf1))
    say()

    a0, f0, _ = results["未校正（argmax 原始機率）"]
    a1, f1v, p1 = results["先驗校正後"]
    say("Accuracy %+.4f，Macro F1 %+.4f。" % (a1 - a0, f1v - f0))
    say()
    say("校正後各類別：")
    say("```")
    say(classification_report(y_te, p1, target_names=TIER_NAMES, digits=3))
    say("```")
    say()
    say("> 校正會改變 argmax，因此指標必然變動。若 Macro F1 下降，代表 SMOTE 原本")
    say("> 就是靠高估少數類別換取 Macro F1；要兩者兼顧應改用類別權重加機率校準。")
    say()


# -- 檢查 2 ------------------------------------------------------------
def check_posthoc_leakage(X, y, features):
    say("## 檢查 2 — 事後變數是否灌水")
    say()
    present = [c for c in POST_HOC if c in features]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X[features], y, test_size=0.2, stratify=y, random_state=42)
    ratio = prior_ratio_from(y_tr)

    variants = [("完整特徵（基準）", features)]
    if "dlc_count" in present:
        variants.append(("移除 dlc_count", [f for f in features if f != "dlc_count"]))
    if present:
        variants.append(("移除全部事後變數（%s）" % ", ".join(present),
                         [f for f in features if f not in present]))

    say("| 特徵集 | 維度 | Accuracy | Macro F1 |")
    say("|---|---|---|---|")
    base = None
    for label, cols in variants:
        _, proba = fit_predict(X_tr[cols], y_tr, X_te[cols])
        acc, mf1, _ = score(y_te, proba, ratio)
        if base is None:
            base = (acc, mf1)
            say("| %s | %d | %.4f | %.4f |" % (label, len(cols), acc, mf1))
        else:
            say("| %s | %d | %.4f (%+.4f) | %.4f (%+.4f) |"
                % (label, len(cols), acc, acc - base[0], mf1, mf1 - base[1]))
    say()
    say("> dlc_count 取自 2025 快照，對 2016 年發行的遊戲累積了 9 年。發行商為賣得好的")
    say("> 遊戲做 DLC，因果是反的。掉幅越大，代表原模型越依賴這個洩漏。")
    say()


# -- 檢查 3 ------------------------------------------------------------
def check_tag_coverage(X, y, features):
    say("## 檢查 3 — 標籤覆蓋率捷徑")
    say()
    tag_cols = [c for c in features if c.startswith("tag_")]
    if not tag_cols or "release_year" not in X.columns:
        say("_缺少 tag_* 欄位或 release_year，跳過。_")
        say()
        return

    has_tag = X[tag_cols].sum(axis=1) > 0
    say("標籤矩陣覆蓋 %s / %s 筆（%.1f%%）"
        % (format(int(has_tag.sum()), ","), format(len(has_tag), ","),
           has_tag.mean() * 100))
    say()

    say("**各發行年份的標籤覆蓋率**")
    say()
    by_year = pd.crosstab(X["release_year"], has_tag, normalize="index")
    counts = X["release_year"].value_counts().sort_index()
    say("| 發行年 | 有標籤 | 無標籤 | 樣本數 |")
    say("|---|---|---|---|")
    for yr in sorted(by_year.index):
        t = by_year.loc[yr].get(True, 0.0)
        say("| %d | %.1f%% | %.1f%% | %s |"
            % (int(yr), t * 100, (1 - t) * 100, format(int(counts[yr]), ",")))
    say()

    say("**標籤覆蓋與銷量等級的關係**")
    say()
    by_tier = pd.crosstab(has_tag, y, normalize="index")
    say("| | " + " | ".join(TIER_NAMES) + " |")
    say("|---|---|---|---|")
    for flag in [True, False]:
        if flag in by_tier.index:
            r = by_tier.loc[flag]
            say("| %s | %s |" % ("有標籤" if flag else "無標籤",
                                 " | ".join("%.1f%%" % (r.get(i, 0.0) * 100)
                                            for i in range(3))))
    say()

    # 只用 has_tag 單一特徵能做到多準
    Xs = pd.DataFrame({"has_tag": has_tag.astype(int).to_numpy()})
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.2, stratify=y, random_state=42)
    _, proba = fit_predict(X_tr, y_tr, X_te)
    acc, mf1, _ = score(y_te, proba, prior_ratio_from(y_tr))
    major = np.bincount(y, minlength=3).max() / len(y)
    say("**只用 `has_tag` 單一特徵**：Accuracy %.4f、Macro F1 %.4f（全猜多數類別為 %.4f）"
        % (acc, mf1, major))
    say()
    say("> 若標籤覆蓋率隨年份陡降、且無標籤者明顯偏 Flop，模型可用「標籤全零」")
    say("> 當作「近年發行」的捷徑，而非真的學到標籤與銷量的關係。")
    say()


# -- 檢查 4 ------------------------------------------------------------
def check_temporal_split(X, y, features):
    say("## 檢查 4 — 時序切分 vs 隨機切分")
    say()
    if "release_year" not in X.columns:
        say("_缺少 release_year，跳過。_")
        say()
        return

    yr = X["release_year"]
    tr_m, te_m = (yr <= 2021).to_numpy(), (yr == 2023).to_numpy()
    if tr_m.sum() == 0 or te_m.sum() == 0:
        say("_年份範圍不足以做時序切分，跳過。_")
        say()
        return

    say("| 切分方式 | 訓練 | 測試 | Accuracy | Macro F1 |")
    say("|---|---|---|---|---|")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X[features], y, test_size=0.2, stratify=y, random_state=42)
    _, proba = fit_predict(X_tr, y_tr, X_te)
    acc_r, f1_r, _ = score(y_te, proba, prior_ratio_from(y_tr))
    say("| 隨機 80/20 | %s | %s | %.4f | %.4f |"
        % (format(len(X_tr), ","), format(len(X_te), ","), acc_r, f1_r))

    Xt, yt = X.loc[tr_m, features], y[tr_m]
    Xs, ys = X.loc[te_m, features], y[te_m]
    _, proba = fit_predict(Xt, yt, Xs)
    acc_t, f1_t, _ = score(ys, proba, prior_ratio_from(yt))
    say("| 時序 ≤2021 → 2023 | %s | %s | %.4f (%+.4f) | %.4f (%+.4f) |"
        % (format(len(Xt), ","), format(len(Xs), ","),
           acc_t, acc_t - acc_r, f1_t, f1_t - f1_r))
    say()
    say("> 資料橫跨 2015–2023 且各年中位數銷量逐年下降，隨機切分等於拿 2023 的資料")
    say("> 預測 2020。時序切分的數字才是模型面對未來遊戲的真實表現。")
    say()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="建模表 CSV")
    ap.add_argument("--meta", default=str(ROOT / "model_meta.json"))
    ap.add_argument("--label", default="tier")
    ap.add_argument("--outdir", default=str(ROOT / "analysis" / "out"))
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    features = meta["selected_features"]
    df = pd.read_csv(args.data)

    missing = [c for c in features + [args.label] if c not in df.columns]
    if missing:
        sys.exit("建模表缺少 %d 個必要欄位：\n  %s%s"
                 % (len(missing), "\n  ".join(missing[:20]),
                    "\n  ..." if len(missing) > 20 else ""))
    if "release_year" not in df.columns:
        print("warning: 沒有 release_year，檢查 3 與 4 會跳過\n", file=sys.stderr)

    y = df[args.label].astype(int).to_numpy()
    X = df.drop(columns=[args.label]).reset_index(drop=True)

    say("# 診斷報告")
    say()
    say("樣本 %s 筆 · 特徵 %d 維 · 類別分布 %s"
        % (format(len(df), ","), len(features),
           " / ".join("%s %s(%.1f%%)" % (n, format(int(c), ","), c / len(y) * 100)
                      for n, c in zip(TIER_NAMES, np.bincount(y, minlength=3)))))
    say()

    check_prior_correction(X, y, features)
    check_posthoc_leakage(X, y, features)
    check_tag_coverage(X, y, features)
    check_temporal_split(X, y, features)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.md").write_text("\n".join(_report_lines), encoding="utf-8")
    print("\n報告已寫入 %s" % (out / "diagnostics.md"))


if __name__ == "__main__":
    main()
