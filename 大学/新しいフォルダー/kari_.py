import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# --- 定数定義 ---
total_minutes = 1440
interval = 5  # 5分刻み
num_intervals = total_minutes // interval  # 288ステップ

modes = [0, 25, 50, 75, 100]  # 電力[%]
kLa_ratios = [0.0, 0.45, 0.67, 0.86, 1.0]  # 相対KLa
kLa_max = 0.0309  # 100%出力時のKLa [1/min]

R_base = 0.067 * 0.67  # 酸素消費ベース値 [mg/L/min]
Cs = 7.75             # 飽和DO [mg/L]
C0 = 5.4              # 初期DO [mg/L]
min_DO = 5.0          # 最低DO制約 [mg/L]

# --- 時間軸作成 ---
time = np.arange(total_minutes + 1)

# --- 酸素消費 R_t の生成（夜間スケール＋給餌＋ノイズあり） ---
R_t = np.full_like(time, R_base, dtype=float)

def create_night_scaling(time, start=660, night_mean=0.9):
    t_mod = time.copy()
    t_mod[t_mod < start] += 1440
    night_idx = np.where((t_mod >= start) & (t_mod < start + 780))[0]
    norm_t = (t_mod[night_idx] - start) / 780
    peak_pos = 0.5
    scale_vals = 1 - 0.2 * (1 - np.abs(norm_t - peak_pos) / peak_pos)
    scale_vals *= (night_mean / np.mean(scale_vals))
    scale = np.ones_like(time, dtype=float)
    scale[night_idx] = scale_vals
    return scale

night_scale = create_night_scaling(time)
R_t *= night_scale

for feed_time in [0, 240, 600]:
    feeding_mask = (time >= feed_time) & (time < feed_time + 30)
    R_t[feeding_mask] *= 1.4

np.random.seed(42)
noise = np.random.normal(0, 0.02, size=len(time))
R_t *= (1 + noise)
# ★修正点1: R_tが負の値にならないようにクリップ
R_t = np.maximum(0, R_t)


# --- モデル構築 ---
model = gp.Model("DO_Optimization_5min")

# 変数: y[t, m] ∈ {0,1}（t番目の5分区間にモードmを使うか）
y = model.addVars(num_intervals, len(modes), vtype=GRB.BINARY, name="y")

# DOの状態変数（5分刻み）
C = model.addVars(num_intervals + 1, vtype=GRB.CONTINUOUS, name="C")
model.addConstr(C[0] == C0)

# 各5分区間に1モードのみ選択
for t5 in range(num_intervals):
    model.addConstr(gp.quicksum(y[t5, m] for m in range(len(modes))) == 1)

# DO遷移式・最低DO制約
for t5 in range(num_intervals):
    kLa_expr = gp.quicksum(y[t5, m] * kLa_ratios[m] for m in range(len(modes))) * kLa_max
    # 5分分の酸素消費の平均を使う（mg/L/min単位で平均）
    R_avg = sum(R_t[t5*interval + i] for i in range(interval)) / interval
    # DO遷移（5分分まとめて計算）
    model.addConstr(
        C[t5+1] == C[t5] + interval * (kLa_expr * (Cs - C[t5]) - R_avg)
    )
    model.addConstr(C[t5+1] >= min_DO)

# 目的関数：電力最小化（5分間の電力×時間の重み=5分なので5倍）
power = gp.quicksum(y[t5, m] * modes[m] * interval for t5 in range(num_intervals) for m in range(len(modes)))
model.setObjective(power, GRB.MINIMIZE)

# --- 最適化実行 ---
model.Params.TimeLimit = 300  # タイムリミット（秒）を適宜設定
model.optimize()

# --- 結果取得・表示 ---
if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
    C_result = [C[i].X for i in range(num_intervals + 1)]
    mode_result = [sum(modes[m] * y[t5, m].X for m in range(len(modes))) for t5 in range(num_intervals)]

    # ★修正点2: R_t_avg_per_interval の準備
    R_t_avg_per_interval = [sum(R_t[t5*interval + i] for i in range(interval)) / interval for t5 in range(num_intervals)]
    R_t_avg_expanded = np.repeat(R_t_avg_per_interval, interval)


    # 時間軸(分)に合わせて伸張（5分間同じモード値を繰り返す）
    mode_result_expanded = np.repeat(mode_result, interval)
    time_expanded = np.arange(total_minutes)

    plt.figure(figsize=(12,6))
    plt.plot(time_expanded, np.interp(time_expanded, np.arange(0, total_minutes+1, interval), C_result), color='red', label='DO (mg/L)')
    plt.axhline(y=min_DO, color='gray', linestyle='--', label='最低DO制約')
    
    # ★修正点3: 酸素消費R_tをプロットに追加
    plt.plot(time_expanded, R_t_avg_expanded * 100, color='green', linestyle=':', alpha=0.7, label='酸素消費 R_avg * 100') # 見やすいようにスケール調整
    
    plt.xlabel('時間 [分] (8:00開始)')
    plt.ylabel('DO (mg/L)')
    plt.title('5分刻みモード選択で最小電力・最低DO維持')
    plt.grid(True)
    plt.legend(loc='upper left')

    ax2 = plt.gca().twinx()
    ax2.plot(time_expanded, mode_result_expanded, color='blue', alpha=0.5, label='出力モード (%)')
    ax2.set_ylabel('出力 (%)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
else:
    print("最適解が得られませんでした。")