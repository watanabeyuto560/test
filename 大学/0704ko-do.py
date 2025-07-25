import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum 
import matplotlib.font_manager as fm
import pandas as pd 

# 日本語フォント設定（Windows Meiryo）
plt.rcParams["font.family"] = "Meiryo"
    
# --- 時間定義 ---
DAY = 1
HOUR = 24
MINUTES = 60
SECOND = 60
DELTA_TIME = 300    # 5分刻み (秒)
total_minutes = DAY * HOUR * MINUTES
interval = DELTA_TIME // 60
num_intervals = total_minutes // interval
time = np.arange(total_minutes + 1)

# --- モードとkLaの設定 ---
modes = [0, 25, 50, 75, 100]
kLa_ratios = [0.0, 0.45, 0.67, 0.86, 1.0]
kLa_max = 0.0309

# 100%時の消費電力 (W)
POWER_100_PERCENT_W = 150 

# --- 酸素関係定数 ---(mg/L)
R_base = 0.067 * 0.90 * 1.1
Cs = 7.75
C0 = 5.2
min_DO = 5.05 # min_DOはグラフ描画の参照用として残すが、シミュレーションには影響しない

R_t = np.full_like(time, R_base, dtype=float)

def create_night_scaling_gaussian(time, night_center=1350, peak_min=0.8, width=180):
    distance = np.abs(((time - night_center + 720) % 1440) - 720)
    scale = 1 - (1 - peak_min) * np.exp(-(distance ** 2) / (2 * width ** 2))
    return scale

R_t *= create_night_scaling_gaussian(time)
for feed_time in [480, 720, 1080]:
    R_t[(time >= feed_time) & (time < feed_time + 30)] *= 1.4
R_t = np.maximum(0, R_t)
R_t_avg_per_interval = [np.mean(R_t[t5*interval:(t5+1)*interval]) for t5 in range(num_intervals)]

# =========================================================================================
# --- PV発電量（日射量から変換・補間）---
# 1時間ごとの日射積算値 (Wh/m^2/h)。これはそのままW/m^2の平均強度として扱えます。
# このデータは例であり、実際の日射データに置き換えてください。
hourly_solar_irradiance_Wh_per_sqm_h = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,      # 0時-7時 (夜間)
    150, 400, 600, 800, 900, 850, 650, 300, # 8時-16時 (日中ピークを調整)
    100, 0, 0, 0, 0, 0, 0         # 17時-23時 (夕方まで考慮)
]) 

# 元のデータポイントの時間 (h)
hourly_time_points_hours = np.arange(len(hourly_solar_irradiance_Wh_per_sqm_h))

# 全シミュレーション期間の5分間隔の時間軸 (h)
simulation_time_points_hours = np.arange(0, total_minutes / MINUTES, interval / MINUTES)

# 日射強度 [W/m^2] を5分値に線形補間
interpolated_irradiance_W_per_sqm = np.interp(
    simulation_time_points_hours,
    hourly_time_points_hours,
    hourly_solar_irradiance_Wh_per_sqm_h # Wh/m^2/h を W/m^2 とみなして補間
)

# PVパネルの公称容量 [kW] - ここをシステムに合わせて設定
PV_PANEL_CAPACITY_KW = 1 # <<< 変更点: PV容量を1kWに設定 >>>

# 標準試験条件での最大日射強度 [W/m^2] (STC)
MAX_REFERENCE_IRRADIANCE_W_PER_SQM = 1000 

# PVシステムからの実際の出力 [W] を計算
# 日射強度に比例してPV容量に応じた出力を計算
pv_power_W_series = interpolated_irradiance_W_per_sqm * (PV_PANEL_CAPACITY_KW * 1000) / MAX_REFERENCE_IRRADIANCE_W_PER_SQM

# PV出力が負にならないように0でクリップ
pv_power_W_series[pv_power_W_series < 0] = 0

assert len(pv_power_W_series) == num_intervals
# =========================================================================================


# --- 蓄電池設定 ---
battery_capacity = 5200      # Wh 
SOC0 = battery_capacity / 2
min_SOC = battery_capacity * 0.3
max_SOC = battery_capacity * 0.9

# --- 充放電レートの最大値 (W) ---
# MAX_CHARGE_RATE_W と MAX_DISCHARGE_RATE_W は、PVの最大出力を考慮して設定
# PV容量が1kWなので、レートもそれに応じた現実的な値に変更することを推奨します。
# 例: 1kWのPVに対して、1kWのインバーターならMAX_CHARGE_RATE_W=1000
# もし0.5Cレート (5200Whバッテリーで2時間でフル充放電) なら 5200Wh/2h = 2600W
# ここはシステムの物理的な最大値に合わせる
MAX_CHARGE_RATE_W = 1000 # 1kWのPVに対しては十分高いレート
MAX_DISCHARGE_RATE_W = 1000 # 1kWのPVに対しては十分高いレート

# --- 追加：充放電効率 ---
CHARGE_EFFICIENCY = 1   # 充電効率 
DISCHARGE_EFFICIENCY = 1 # 放電効率

# --- モデル構築 ---
model = Model("DO_Optimization_PV_Only")
y = model.addVars(num_intervals, len(modes), vtype=GRB.BINARY)
C = model.addVars(num_intervals + 1, vtype=GRB.CONTINUOUS)
SOC = model.addVars(num_intervals + 1, vtype=GRB.CONTINUOUS)

# 追加：充放電量を表す変数
charge_W = model.addVars(num_intervals, vtype=GRB.CONTINUOUS, name="charge_W")
discharge_W = model.addVars(num_intervals, vtype=GRB.CONTINUOUS, name="discharge_W")
# 追加：同時に充放電しないための補助バイナリ変数
is_charging = model.addVars(num_intervals, vtype=GRB.BINARY, name="is_charging")
# 追加：余剰電力の変数
excess_power_wh = model.addVars(num_intervals, vtype=GRB.CONTINUOUS, name="excess_power_wh")


model.addConstr(C[0] == C0)
model.addConstr(SOC[0] == SOC0) 

for t5 in range(num_intervals):
    model.addConstr(quicksum(y[t5, m] for m in range(len(modes))) == 1)
    kLa_expr = quicksum(y[t5, m] * kLa_ratios[m] for m in range(len(modes))) * kLa_max
    R_avg = R_t_avg_per_interval[t5]
    model.addConstr(C[t5+1] == C[t5] + interval * (kLa_expr * (Cs - C[t5]) - R_avg))
    model.addConstr(C[t5+1] >= min_DO)

    power_consume_W_expr = quicksum(y[t5, m] * modes[m] * POWER_100_PERCENT_W / 100 for m in range(len(modes)))
    power_consume_wh = power_consume_W_expr * (interval / 60) # 負荷消費量 (Wh)
    
    pv_wh_interval = pv_power_W_series[t5] * (interval / 60) # PV発電量 (Wh)

    # SOC変化式はt5=0の場合も含め、ループ内で前時刻のSOC[t5]から計算する形式に統一
    model.addConstr(SOC[t5+1] == SOC[t5] + (charge_W[t5] * (interval / 60)) * CHARGE_EFFICIENCY - (discharge_W[t5] * (interval / 60)) / DISCHARGE_EFFICIENCY)

    # 蓄電池SOCの上限下限制約はそのまま
    model.addConstr(SOC[t5+1] >= min_SOC)
    model.addConstr(SOC[t5+1] <= max_SOC)

    # 電力バランス制約 (余剰電力の概念を追加)
    model.addConstr(pv_wh_interval + (discharge_W[t5] * (interval / 60)) == power_consume_wh + (charge_W[t5] * (interval / 60)) + excess_power_wh[t5],
                    name=f"power_balance_{t5}")

    # 充放電レート制限をW単位で適用
    model.addConstr(charge_W[t5] >= 0, name=f"charge_non_negative_{t5}")
    model.addConstr(discharge_W[t5] >= 0, name=f"discharge_non_negative_{t5}")
    model.addConstr(excess_power_wh[t5] >= 0, name=f"excess_power_non_negative_{t5}") # 余剰電力も非負
    
    model.addConstr(charge_W[t5] <= MAX_CHARGE_RATE_W, name=f"max_charge_rate_{t5}")
    model.addConstr(discharge_W[t5] <= MAX_DISCHARGE_RATE_W, name=f"max_discharge_rate_{t5}")
    
    # 同時に充放電しない（線形化MIP制約）
    MAX_M_VALUE_W = max(MAX_CHARGE_RATE_W, MAX_DISCHARGE_RATE_W, POWER_100_PERCENT_W, pv_power_W_series.max()) + 100 # 十分大きな値
    model.addConstr(charge_W[t5] <= MAX_M_VALUE_W * is_charging[t5], name=f"charge_implies_charging_mode_{t5}")
    model.addConstr(discharge_W[t5] <= MAX_M_VALUE_W * (1 - is_charging[t5]), name=f"discharge_implies_discharging_mode_{t5}")

model.addConstr(SOC[num_intervals] >= SOC0 - 0.01 * battery_capacity, name="final_SOC_lower_bound")

# --- DOの変化を緩やかにする制約 ---
for t in range(num_intervals - 12):
    model.addConstr(C[t + 12] - C[t] <= 0.7)
    model.addConstr(C[t] - C[t + 12] <= 0.7)
for t1 in range(num_intervals - 12, num_intervals):
    for t2 in range(t1 + 1, num_intervals + 1):
        model.addConstr(C[t2] - C[t1] <= 0.4)
        model.addConstr(C[t1] - C[t2] <= 0.4)

# --- 目的関数：平均DO最大化 (AND 余剰電力最小化) ---
total_excess_power_wh = quicksum(excess_power_wh[t5] for t5 in range(num_intervals))
penalty_weight_for_excess_power = 1e-5 # ここで重みを設定

avg_DO = quicksum(C[t5] for t5 in range(1, num_intervals + 1)) / num_intervals
model.setObjective(avg_DO - penalty_weight_for_excess_power * total_excess_power_wh, GRB.MAXIMIZE)

model.Params.TimeLimit = 60
model.optimize()

# --- 固定モードの設定（比較用） ---
mode_array = np.full(num_intervals, 100) 
num_stops = 10
stop_segment_length_steps = num_intervals / num_stops 

for i in range(num_stops):
    stop_end_index = int(round((i + 1) * stop_segment_length_steps)) - 1
    if 0 <= stop_end_index < num_intervals:
        mode_array[stop_end_index] = 0
    elif i == num_stops - 1 and num_intervals > 0:
        mode_array[num_intervals - 1] = 0


# 固定モードでのDOシミュレーション
C_fixed = np.zeros(num_intervals + 1)
C_fixed[0] = C0
for t5 in range(num_intervals):
    mode = mode_array[t5]
    kLa = kLa_ratios[modes.index(mode)] * kLa_max
    R_avg = R_t_avg_per_interval[t5]
    C_next = C_fixed[t5] + interval * (kLa * (Cs - C_fixed[t5]) - R_avg)
    C_fixed[t5+1] = C_next 

# 固定モードの消費電力計算（W）- POWER_100_PERCENT_Wを使用
power_consume_fixed_series_W = [mode * POWER_100_PERCENT_W / 100 for mode in mode_array]

# 固定モードのSOC計算（簡易）- SOC上下限考慮なし
SOC_fixed = np.zeros(num_intervals + 1)
SOC_fixed[0] = SOC0
for t5 in range(num_intervals):
    pv_wh_current = pv_power_W_series[t5] * (interval / 60)
    power_consume_fixed_wh_current = power_consume_fixed_series_W[t5] * (interval / 60)
    
    net_power_W = pv_power_W_series[t5] - power_consume_fixed_series_W[t5] # W単位での正味電力
    
    charge_fixed_W = 0
    discharge_fixed_W = 0

    if net_power_W > 0: # PV > 負荷 -> 充電
        charge_fixed_W = min(net_power_W, MAX_CHARGE_RATE_W) 
        
    else: # PV < 負荷 -> 放電
        discharge_fixed_W = min(abs(net_power_W), MAX_DISCHARGE_RATE_W) 

    SOC_fixed[t5+1] = SOC_fixed[t5] + (charge_fixed_W * (interval / 60)) * CHARGE_EFFICIENCY - (discharge_fixed_W * (interval / 60)) / DISCHARGE_EFFICIENCY
    
    # 最終的なSOCが物理的な制約内に収まるように調整 (厳密にはモデル構築時に行われるが、固定モードでは手動で)
    # SOC_fixed[t5+1] = max(min_SOC, min(SOC_fixed[t5+1], max_SOC)) 


# --- 結果表示 ---
# CSV保存先ディレクトリを定義
output_dir = r"C:\Users\fukui\PVSOC" 

if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
    print("--- 最適化結果の処理を開始します ---") 
    C_opt = [C[i].X for i in range(num_intervals + 1)]
    mode_result = [sum(modes[m] * y[t5, m].X for m in range(len(modes))) for t5 in range(num_intervals)]
    SOC_result = [SOC[i].X for i in range(num_intervals + 1)]
    
    # 充放電量の結果をW単位で取得
    charge_series_W = [charge_W[t5].X for t5 in range(num_intervals)]
    discharge_series_W = [discharge_W[t5].X for t5 in range(num_intervals)]

    power_consume_series_W = [sum(y[t5, m].X * modes[m] * POWER_100_PERCENT_W / 100 for m in range(len(modes)))
                                  for t5 in range(num_intervals)]
    
    # 余剰電力の取得 (Wh → W)
    excess_power_series_W = [excess_power_wh[t5].X / (interval / 60) for t5 in range(num_intervals)]


    total_consumption_wh = sum([p_w * (interval / 60) for p_w in power_consume_series_W])
    total_pv_wh = sum([p_w * (interval / 60) for p_w in pv_power_W_series]) 
    total_consumption_fixed_wh = sum([p_w * (interval / 60) for p_w in power_consume_fixed_series_W])


    print(f"▶ 総消費電力量 (最適化): {total_consumption_wh:.1f} Wh")
    print(f"▶ 総PV出力量 (最適化):      {total_pv_wh:.1f} Wh")
    print(f"▶ 最終SOC (最適化):        {SOC_result[-1]:.1f} Wh（初期: {SOC0:.1f} Wh）")
    print(f"▶ 平均DO (最適化):          {np.mean(C_opt):.3f} mg/L")
    print(f"▶ 総消費電力量 (固定モード): {total_consumption_fixed_wh:.1f} Wh")
    print(f"▶ 平均DO (固定モード50/75%): {np.mean(C_fixed):.3f} mg/L")

    time_hours = np.arange(num_intervals + 1) * interval / 60

    plt.figure(figsize=(12, 12)) 
    
    plt.subplot(5,1,1)
    plt.plot(time_hours, C_opt, label='最適化 DO [mg/L]', color='blue')
    plt.plot(time_hours, C_fixed, label='固定モード DO [mg/L]', color='orange', linestyle='--')
    plt.axhline(min_DO, color='r', linestyle='--', label='Min DO')
    plt.ylabel('DO [mg/L]')
    plt.legend()
    plt.grid(True)

    plt.subplot(5,1,2)
    plt.step(time_hours[:-1], mode_result, where='post', label='最適化 Mode [%]', color='blue')
    plt.step(time_hours[:-1], mode_array, label='固定モード Mode [%]', color='orange', linestyle='--')
    plt.ylabel('Aeration Mode [%]')
    plt.legend()
    plt.yticks(modes)
    plt.ylim(-5, 105)
    plt.grid(True)

    plt.subplot(5,1,3)
    plt.plot(time_hours, np.array(SOC_result)/battery_capacity, label='最適化 SOC', color='blue')
    plt.plot(time_hours, SOC_fixed / battery_capacity, label='固定モード SOC', color='orange', linestyle='--')
    plt.ylabel('SOC [%]')
    plt.legend()
    plt.grid(True)

    plt.subplot(5,1,4)
    plt.plot(time_hours[:-1], charge_series_W, label='最適化 Charge (W)', color='cyan')
    plt.plot(time_hours[:-1], -np.array(discharge_series_W), label='最適化 Discharge (W)', color='magenta', linestyle='--') 
    plt.ylabel('Battery Power [W]')
    plt.legend()
    plt.grid(True)


    plt.subplot(5,1,5) 
    plt.plot(time_hours[:-1], pv_power_W_series, label='PV Generation (W)', linestyle='--', color='green') 
    plt.step(time_hours[:-1], power_consume_series_W, where='post', label='最適化 Power Consume (W)', color='blue')
    plt.step(time_hours[:-1], power_consume_fixed_series_W, label='固定モード Power Consume (W)', linestyle='--', color='orange')
    plt.plot(time_hours[:-1], excess_power_series_W, label='Excess Power (W)', color='red', linestyle=':') # 余剰電力を追加
    plt.ylabel('Power [W]')
    plt.xlabel('Time [h]')
    plt.legend()
    plt.grid(True) 

    plt.tight_layout()
    
    # グラフをファイルに保存
    graph_filename = os.path.join(output_dir, f"simulation_graph_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    try:
        plt.savefig(graph_filename)
        print(f"グラフが {graph_filename} に保存されました。")
    except Exception as e:
        print(f"エラー: グラフ保存中に問題が発生しました: {e}")
    
    plt.show() # グラフを表示 (この行の後にCSV保存が実行される)

    # CSV保存処理
    print(f"--- CSV保存処理を開始します ---") 
    print(f"保存先ディレクトリ: {output_dir}") 
    try:
        os.makedirs(output_dir, exist_ok=True) 
        print("ディレクトリの作成/存在確認が完了しました。") 
    except Exception as e:
        print(f"エラー: ディレクトリ作成中に問題が発生しました: {e}") 
        print("CSV保存処理を中断します。")
        # exit() # ディレクトリ作成失敗時でもスクリプトを中断しないようにコメントアウト (必要に応じて戻す)

    # 最適化結果
    results_opt = {
        'Time (h)': time_hours[:-1], 
        'PV Generation (W)': pv_power_W_series,
        'SOC (Wh)_Optimal': [soc_val for soc_val in SOC_result[:-1]], 
        'SOC (%)_Optimal': [soc_val / battery_capacity for soc_val in SOC_result[:-1]], 
        'Charge (W)_Optimal': charge_series_W,
        'Discharge (W)_Optimal': [-d for d in discharge_series_W], 
        'Optimal Mode (%)': mode_result,
        'Optimal DO (mg/L)': C_opt[:-1], 
        'Oxygen Consumption [mg/L/5min]': R_t_avg_per_interval,
        'Optimal Power Consume (W)': power_consume_series_W,
        'Excess Power (W)_Optimal': excess_power_series_W
    }
    df_opt = pd.DataFrame(results_opt)
    df_opt.set_index('Time (h)', inplace=True)
    print(f"df_opt の形状: {df_opt.shape}, 空かどうか: {df_opt.empty}") 


    # 固定モード結果
    charge_fixed_series_W = []
    discharge_fixed_series_W = []
    excess_fixed_series_W = [] 

    temp_SOC_fixed = SOC0 
    for t5 in range(num_intervals):
        pv_wh_current = pv_power_W_series[t5] * (interval / 60)
        power_consume_fixed_wh_current = power_consume_fixed_series_W[t5] * (interval / 60)
        
        net_power_W_calc = pv_power_W_series[t5] - power_consume_fixed_series_W[t5] # W単位での正味電力
        
        charge_fixed_W_calc = 0
        discharge_fixed_W_calc = 0
        excess_fixed_W_calc = 0

        if net_power_W_calc > 0: # PV > 負荷 -> 充電
            charge_fixed_W_calc = min(net_power_W_calc, MAX_CHARGE_RATE_W)
            
            if net_power_W_calc > charge_fixed_W_calc: # もし充電レートで吸収しきれないPV余剰があればexcessになる
                excess_fixed_W_calc = net_power_W_calc - charge_fixed_W_calc
        else: # PV < 負荷 -> 放電
            discharge_fixed_W_calc = min(abs(net_power_W_calc), MAX_DISCHARGE_RATE_W)

        temp_SOC_fixed = temp_SOC_fixed + (charge_fixed_W_calc * (interval / 60)) * CHARGE_EFFICIENCY - (discharge_fixed_W_calc * (interval / 60)) / DISCHARGE_EFFICIENCY
        
        charge_fixed_series_W.append(charge_fixed_W_calc)
        discharge_fixed_series_W.append(discharge_fixed_W_calc)
        excess_fixed_series_W.append(excess_fixed_W_calc)


    results_fixed = {
        'Time (h)': time_hours[:-1],
        'PV Generation (W)_Fixed': pv_power_W_series, 
        'SOC (Wh)_Fixed': [soc_val for soc_val in SOC_fixed[:-1]],
        'SOC (%)_Fixed': [soc_val / battery_capacity for soc_val in SOC_fixed[:-1]],
        'Charge (W)_Fixed': charge_fixed_series_W, 
        'Discharge (W)_Fixed': [-d for d in discharge_fixed_series_W], 
        'Fixed Mode (%)': mode_array,
        'Fixed DO (mg/L)': C_fixed[:-1],
        'Oxygen Consumption [mg/L/5min]_Fixed': R_t_avg_per_interval, 
        'Fixed Power Consume (W)': power_consume_fixed_series_W,
        'Excess Power (W)_Fixed': excess_fixed_series_W 
    }
    df_fixed = pd.DataFrame(results_fixed)
    df_fixed.set_index('Time (h)', inplace=True)
    print(f"df_fixed の形状: {df_fixed.shape}, 空かどうか: {df_fixed.empty}") 


    # データを結合して一つのCSVに保存
    try:
        df_combined = df_opt.merge(df_fixed, left_index=True, right_index=True)
        print(f"df_combined の形状: {df_combined.shape}, 空かどうか: {df_combined.empty}") 
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = os.path.join(output_dir, f"simulation_results_{timestamp}.csv")
        print(f"ファイル名: {output_filename}") 
        
        df_combined.to_csv(output_filename)
        print(f"--- 結果が {output_filename} に正常に保存されました。 ---") 
    except Exception as e:
        print(f"エラー: CSVファイル保存中に問題が発生しました: {e}") 
        print("CSV保存処理を中断します。")

else:
    print("最適解が得られませんでした。")
    model.computeIIS()
    model.write(os.path.join(output_dir, "model_infeasible.ilp"))