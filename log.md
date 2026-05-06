# Development Log - Cliff Walking (Q-Learning vs SARSA)

## [2026-05-03]

- Initialized project.
- Planned the structure: `cliff_walking.py`, `agents.py`, `train.py`, `analysis.py`.
- Implemented `CliffWalkingEnv` in `cliff_walking.py`.
- Implemented `QLearningAgent` and `SARSAAgent` in `agents.py`.
- Installed `numpy` and `matplotlib`.
- Ran training script `train.py`, saved results to `results.pkl`.
- Started implementing `analysis.py` for visualization and report.

## [2026-05-03] — 改進與完善

### 評估發現的不足

- 到達終點獎勵改為 -1（每步一律 -1）
- 只有單次實驗，缺乏統計意義
- 獎勵曲線缺少信賴區間
- 路徑視覺化不夠清晰
- 報告缺乏量化分析數據
- 缺少 Q 值熱力圖

### 修改內容

1. **`cliff_walking.py`**: 增加常數定義、docstring、到達終點獎勵改為 -1、cliff 改為 set 提升查找效率
2. **`agents.py`**: 增加 `get_greedy_action()` 方法、完善 docstring
3. **`train.py`**: 新增 `run_multiple_experiments()` 支援 30 次獨立實驗、統計摘要
4. **`analysis.py`**: 全面重寫
   - 獎勵曲線改為 mean ± std 信賴區間（raw + smoothed 雙圖）
   - 路徑視覺化加入 greedy policy 箭頭、顏色圖例、步數標示
   - 新增 Q 值熱力圖 (`q_value_heatmap_qlearning.png`, `q_value_heatmap_sarsa.png`)
   - 新增量化穩定性分析（CV、收斂速度）
5. **`REPORT.md`**: 全面改寫
   - 加入 30 次實驗的統計數據
   - 加入 Q 值分佈分析
   - 量化穩定性比較表
   - 更詳盡的理論討論
   - 應用場景建議表

### 重新訓練結果（30 次 × 500 回合）

- Q-Learning 最後 50 回合平均: -48.34 ± 9.47
- SARSA 最後 50 回合平均: -23.49 ± 3.66
- SARSA 更穩定（CV: 0.84 vs 1.41）
- Q-Learning greedy 路徑: 13 步（懸崖邊緣）
- SARSA greedy 路徑: 17 步（最頂部安全路線）

## [2026-05-06]

- 解釋了 `analysis.py` 中 `reward_plot.png` 左右兩圖的差異：左圖為原始數據（Raw），展示訓練過程的隨機波動；右圖為平滑數據（Smoothed, window=20），用於更清晰地觀察收斂趨勢。
