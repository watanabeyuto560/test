import numpy as np
import matplotlib.pyplot as plt

# --- 定数定義 ---
total_minutes = 1440
interval = 5  # 5分刻み
num_intervals = total_minutes // interval  # 288ステップ

# kLa_ratios や modes は最適化部分を削除するため不要だが、ここでは仮に置いておく
# modes = [0, 25, 50, 75, 100]  # 電力[%]
# kLa_ratios = [0.0, 0.45, 0.67, 0.86, 1.0]  # 相対KLa
# kLa_max = 0.0309  # 100%出力時のKLa [1/min]

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
# R_tが負の値にならないようにクリップ
R_t = np.maximum(0, R_t)


# --- DO濃度の手動シミュレーション（消費のみ） ---
# KLaは0（酸素供給なし）と仮定し、純粋な消費の影響を見る
C_sim = np.zeros(num_intervals + 1)
C_sim[0] = C0

for t5 in range(num_intervals):
    R_avg = sum(R_t[t5*interval + i] for i in range(interval)) / interval
    # 酸素供給 (kLa_expr) を0として計算
    C_sim[t5+1] = C_sim[t5] + interval * (0.0 * (Cs - C_sim[t5]) - R_avg)
    # 最低DO制約はここでは適用しない（消費のみを見るため）
    # C_sim[t5+1] = max(C_sim[t5+1], min_DO) # 必要であれば加える

# --- 結果取得・表示 ---
# 時間軸(分)に合わせて伸張
time_expanded = np.arange(total_minutes)
# DOシミュレーション結果を時間軸に合わせて伸張
C_sim_expanded = np.interp(time_expanded, np.arange(0, total_minutes+1, interval), C_sim)

# R_t の平均値を計算し、プロット用に伸張
R_t_avg_per_interval = [sum(R_t[t5*interval + i] for i in range(interval)) / interval for t5 in range(num_intervals)]
R_t_avg_expanded = np.repeat(R_t_avg_per_interval, interval)


plt.figure(figsize=(12,6))
plt.plot(time_expanded, C_sim_expanded, color='red', label='DO (mg/L) - 消費のみ')
plt.axhline(y=min_DO, color='gray', linestyle='--', label='最低DO制約') # 参考として表示

# 酸素消費R_tをプロットに追加 (見やすいようにスケール調整)
# DO濃度と同じY軸に表示するために、適切なスケールを探る必要があるかもしれません
# ここでは暫定的にR_baseの最大値と比較してスケールしています
max_R_t_avg = np.max(R_t_avg_per_interval)
scale_factor_for_plot = Cs / max_R_t_avg if max_R_t_avg > 0 else 100 # DOスケールに合わせる
plt.plot(time_expanded, R_t_avg_expanded * scale_factor_for_plot, color='green', linestyle=':', alpha=0.7, label=f'酸素消費 R_avg * {scale_factor_for_plot:.0f}')


plt.xlabel('時間 [分] (8:00開始)')
plt.ylabel('DO (mg/L)')
plt.title('DO濃度シミュレーション（酸素消費のみの影響）')
plt.grid(True)
plt.legend(loc='upper left')

# もし酸素消費を別のY軸で表示したい場合（推奨）
# ax2 = plt.gca().twinx()
# ax2.plot(time_expanded, R_t_avg_expanded, color='green', linestyle=':', alpha=0.7, label='酸素消費 R_avg [mg/L/min]')
# ax2.set_ylabel('酸素消費 [mg/L/min]')
# ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()