import numpy as np
import matplotlib.pyplot as plt

# グラフの見た目の標準化
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# 1. 基本的な時間減衰型シャノン情報量の定義
def time_decaying_information(p, t, lambda_param):
    """
    I(p, t) = -log2(p) * exp(-lambda * t)
    """
    shannon_info = -np.log2(p)
    return shannon_info * np.exp(-lambda_param * t)

# --- Figure 1: 確率の違いによる情報量の時間変化 ---
def plot_figure1():
    t = np.linspace(0, 10, 500)
    probabilities = [0.01, 0.05, 0.2, 0.5]
    lambda_param = 0.5  # 固定減衰率

    plt.figure(figsize=(8, 5))
    for p in probabilities:
        info = time_decaying_information(p, t, lambda_param)
        plt.plot(t, info, label=f'Probability p = {p} (Initial I = {-np.log2(p):.2f} bits)')

    plt.title('Figure 1: Time-Decaying Shannon Information for Different Event Probabilities', fontsize=12)
    plt.xlabel('Time t', fontsize=11)
    plt.ylabel('Information Value I(p, t) [bits]', fontsize=11)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()  # <--- 保存せずに表示する

# --- Figure 2: 減衰率 (lambda) によるカテゴリ別の減衰速度の比較 ---
def plot_figure2():
    t = np.linspace(0, 10, 500)
    p = 0.05  # レア度を固定
    
    # カテゴリ別の減衰率
    categories = {
        'SNS Trend (High Decay, λ=1.5)': 1.5,
        'Daily Epidemiological/News Data (Medium Decay, λ=0.5)': 0.5,
        'Academic Theorem / Static Knowledge (Low Decay, λ=0.02)': 0.02
    }

    plt.figure(figsize=(8, 5))
    for label, lam in categories.items():
        info = time_decaying_information(p, t, lam)
        plt.plot(t, info, label=f'{label}')

    plt.title('Figure 2: Information Value Decay Across Different Contexts', fontsize=12)
    plt.xlabel('Time t', fontsize=11)
    plt.ylabel('Information Value I(p, t) [bits]', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.show()  # <--- 保存せずに表示する

# --- Figure 3: 情報の定期更新 (Periodic Information Updates) モデル ---
def plot_figure3():
    t = np.linspace(0, 10, 1000)
    lambda_param = 0.8
    update_interval = 2.0  # 2単位時間ごとに新しい情報が入る
    
    # 時間変化する情報量ベクトル
    info_stream = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        # 最後に更新された時刻からの経過時間
        time_since_last_update = ti % update_interval
        # イベント確率
        p = 0.05
        info_stream[i] = time_decaying_information(p, time_since_last_update, lambda_param)

    plt.figure(figsize=(9, 5))
    plt.plot(t, info_stream, color='crimson', label='Information Stream I(t)')
    for update_time in np.arange(0, 10, update_interval):
        plt.axvline(update_time, color='gray', linestyle=':', alpha=0.7)
    
    plt.title('Figure 3: Periodic Information Renewal Stream', fontsize=12)
    plt.xlabel('Time t', fontsize=11)
    plt.ylabel('Current Information Value [bits]', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.show()  # <--- 保存せずに表示する

if __name__ == '__main__':
    print("グラフを表示します。1つ目のウィンドウを閉じると、次のグラフが表示されます。")
    plot_figure1()
    plot_figure2()
    plot_figure3()