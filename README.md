# SSC Week 10 — Multi-Vessel Gathering

這是準備併入 `SSC_multiboat_week10` Branch 的第二階段內容。第一階段 `mvg.py` 仍使用你原本的版本；請將本資料夾中的 `stage2_analyze.py` 放在專案根目錄，並將 `reports/week10/` 一併加入既有 repo。

## 重現分析

必須先有第一階段產生的三個檔案：

```powershell
python stage2_analyze.py --candidates "outputs/week10_part003/gathering_candidates.csv" --summary "outputs/week10_part003/eda_summary.json" --cells "outputs/week10_part003/top_spatial_cells.csv" --out "outputs/stage2"
```

安裝需求：Python、matplotlib。將 Stage 2 腳本與原先專案的 `requirements.txt` 一起保留。

## 哪些可以提交

- 程式 `stage2_analyze.py`、程式使用說明、`reports/week10/` 中已去除 MMSI 的摘要及三張圖可提交。
- 不要提交原始 AIS CSV、`ais_staging.sqlite`、`gathering_candidates.csv` 或 `selected_cases_private.csv`（含 MMSI）。
- 若現有 `.gitignore` 尚未忽略 `outputs/`、`*.sqlite*`、`*.csv`，請補上相應規則，並注意已被 Git 追蹤的檔案不會因新增 ignore 而自動取消追蹤。

## 報告引用注意

386,230 為五分鐘群集快照候選；2,842 為同一船群連續觀測六個以上時間窗的序列，不代表 2,842 起真正異常事件。案例尚需比對海岸線、港口與原始 AIS 航跡。
