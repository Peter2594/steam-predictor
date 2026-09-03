# Steam 遊戲上市成功預測

用**上市前就取得的資訊**（定價、類型、標籤、成就數、DLC 數、語言數、發行月份），把一款 Steam 遊戲的銷量預測成三個等級：Flop（滯銷）、Normal（回本）、Blockbuster（爆款）。

附一個 Streamlit demo，可以直接調參數看預測怎麼變。

> 國立臺灣大學 · 製造數據科學期末專案（第五組）
> 周靖蓉、林聿平、陳品程、葉家華 — 授課教師：李家岩

---

## 快速開始

```bash
pip install -r requirements.txt
streamlit run app_steam.py
```

或直接在 GitHub Codespaces 開啟 — `.devcontainer/` 已設定好，開起來會自動裝套件並啟動 app。

---

## Model card

| 項目 | 內容 |
|---|---|
| 任務 | 三分類（Flop / Normal / Blockbuster） |
| 演算法 | XGBoost（`n_estimators=300`, `max_depth=7`, `learning_rate=0.05`） |
| 特徵 | 92 維經 LassoCV 篩選後保留 **60** 維 |
| 標籤 | 銷量百分位切分（Model B：50% / 40% / 10%），以 `estimated_owners` 上界為基準 |
| 類別不平衡 | 訓練集套用 SMOTE (k=5) 平衡為 1:1:1，測試集保持原始分布 |
| 樣本 | 2015–2023 年發行，有效樣本 60,814 筆；測試集 12,163 筆 |

### 表現

| 指標 | 值 |
|---|---|
| Accuracy | 67.1% |
| Macro F1 | 0.534 |

| 類別 | Precision | Recall | F1 | 樣本數 |
|---|---|---|---|---|
| Flop | 0.80 | 0.80 | **0.80** | 8,610 |
| Normal | 0.30 | 0.29 | **0.29** | 2,400 |
| Blockbuster | 0.48 | 0.54 | **0.51** | 1,153 |

比較過的其他模型：LightGBM（Acc 0.676 / Macro F1 0.532，訓練快 3 倍）、Random Forest（0.645 / 0.522）、LinearSVC（0.588 / 0.440）。

### Top 10 重要特徵

`price` › `lang_count` › `achievements` › `release_month` › `dlc_count` › `screenshot_count` › `is_multiplayer` › `tag_Action` › `genre_Indie` › `tag_RPG`

---

## 已知限制

這些是實際存在的問題，不是客套話：

- **Normal 類別基本上預測不準**（F1 = 0.29）。中間市場的界線本來就模糊，而且標籤是在一個高度並列的分桶變數（`estimated_owners` 是 `"20000 - 50000"` 這種字串區間）上切百分位，Tier 1 / Tier 2 的邊界很可能落在同一個桶內。這一題也許更適合做序數迴歸，或直接做「進不進前 10%」的二元分類。
- **驗證用隨機切分，不是時序切分**。資料橫跨 2015–2023，而各年度的中位數銷量逐年下降；隨機切分等於拿 2023 的資料去預測 2020，指標會偏樂觀。
- **顯示的機率經過先驗校正**。訓練時 SMOTE 把三類拉成 1:1:1，模型輸出帶著均勻先驗，直接顯示會系統性高估爆款。App 在推論時乘回市場先驗（70.8 / 19.7 / 9.5）再正規化 — 見 `model_meta.json` 的 `train_prior` / `market_prior`。這只挪動機率、不改變類別排序，因此不影響上面的 Accuracy 與 Macro F1。
- **SMOTE 用在 one-hot 稀疏特徵上並不理想**，插值會產生 `tag_Action = 0.37` 這種不存在的樣本。改用 class weight 或 SMOTE-NC 會更合理。
- **只有上市前特徵**，捕捉不到口碑傳播、社群效應、遊戲品質本身。現象級爆款本來就在模型的預測範圍之外。評論數、遊玩時數、Metacritic 分數這類後驗指標與銷量的相關性遠高於前置特徵，但納入即構成資料洩漏，因此全部排除。

---

## 資料來源

Kaggle — *Steam Games Dataset*，使用其中的 `games_march2025_cleaned.csv`。

> 

篩選條件：發行日期介於 2015-01-01 至 2023-12-31，排除定價或銷量區間缺失的樣本。

---

## 檔案

```
app_steam.py        Streamlit demo
model_xgb_clf.pkl   訓練好的 XGBoost 模型
model_meta.json     特徵清單、genre/tag 對照表、先驗
requirements.txt    相依套件
runtime.txt         Python 版本（Streamlit Cloud 用）
.devcontainer/      Codespaces 設定
```

**訓練程式碼目前不在這個 repo 裡** — 清洗、百分位標註、Lasso 篩選、SMOTE、四模型比較與 SHAP 分析都還在本機的 notebook。在補進來之前，這個 repo 是不可重現的。

---

## 驗證環境

模型與 app 在以下版本確認可正常載入與預測：

```
python 3.10 · streamlit 1.49.1 · pandas 2.2.3 · numpy 1.26.4
joblib 1.5.3 · xgboost 3.2.0 · scikit-learn 1.7.2
```

`requirements.txt` 目前沒有鎖版本。pickle 對 xgboost / scikit-learn 的版本敏感，日後套件升級有可能載入失敗；比較穩的作法是改存 XGBoost 原生格式（`clf.save_model("model.json")`），既不受版本影響也避免 pickle 的安全性疑慮。
