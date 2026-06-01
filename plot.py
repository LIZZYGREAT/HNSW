import matplotlib.pyplot as plt
import numpy as np
import os

# 确保输出目录存在
if not os.path.exists('results'):
    os.makedirs('results')

# ==========================================
# 0. 基础数据准备
# ==========================================
algorithms = ['FlatScan', 'HNSW (2D)', 'HNSW (Linear)', 'HNSW (Parallel)']
colors = ['#e63946', '#457b9d', '#1d3557', '#2a9d8f']

# 基础性能数据
qps_data = [93.98, 1234.57, 1176.47, 13333.33]
latency_data = [10640.97, 808.04, 849.57, 510.00]
recall_data = [1.0000, 0.9965, 0.9965, 0.9955]

# 建图时间数据
build_algos = ['HNSW (2D)', 'HNSW (Linear)', 'HNSW (Parallel)']
build_time_data = [80261.00, 74091.00, 5582.00]
build_colors = ['#457b9d', '#1d3557', '#2a9d8f']


def plot_base_bar_charts():
    # --- 图 1：吞吐量 (QPS) 对比 ---
    plt.figure(figsize=(8, 6), dpi=300)
    bars = plt.bar(algorithms, qps_data, color=colors, edgecolor='black', zorder=3)
    plt.yscale('log') # QPS 差异大，使用对数坐标
    plt.title('Throughput Comparison (QPS)', fontsize=14)
    plt.ylabel('Queries Per Second (Log Scale)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.1, f"{int(yval):,}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/throughput_qps.png')
    plt.close()

    # --- 图 2：平均查询延迟对比 ---
    plt.figure(figsize=(8, 6), dpi=300)
    bars = plt.bar(algorithms, latency_data, color=colors, edgecolor='black', zorder=3)
    plt.title('Average Query Latency (us)', fontsize=14)
    plt.ylabel('Latency (microseconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 100, f"{int(yval):,}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/query_latency.png')
    plt.close()

    # --- 图 3：建图耗时对比 ---
    plt.figure(figsize=(8, 6), dpi=300)
    bars = plt.bar(build_algos, build_time_data, color=build_colors, edgecolor='black', width=0.6, zorder=3)
    plt.title('Index Build Time Comparison (ms)', fontsize=14)
    plt.ylabel('Time (milliseconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1000, f"{int(yval):,}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/build_time.png')
    plt.close()

# ==========================================
# 2. 多线程 Scaling 测试数据并绘图
# ==========================================
def plot_scaling_results(filepath):
    threads = []
    build_times = []
    qps = []
    latencies = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()[9:] 
            for line in lines:
                parts = line.split()
                if len(parts) >= 6:
                    threads.append(int(parts[0]))
                    build_times.append(float(parts[1]))
                    qps.append(float(parts[3]))
                    latencies.append(float(parts[4]))
    except FileNotFoundError:
        print(f"[警告] 找不到文件 {filepath}，多线程扩展性图表将被跳过。")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    fig.suptitle('HNSW Multi-thread Scaling Performance', fontsize=16)

    # 图 2.1：建图耗时
    axes[0].plot(threads, build_times, marker='o', markersize=8, linestyle='-', color='#2c5d87', linewidth=2)
    axes[0].set_title('Index Build Time vs Threads')
    axes[0].set_xlabel('Number of Threads')
    axes[0].set_ylabel('Build Time (ms)')
    axes[0].set_xticks(threads)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    for i, txt in enumerate(build_times):
        axes[0].annotate(f"{txt:.0f}", (threads[i], build_times[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 图 2.2：吞吐量 QPS
    axes[1].plot(threads, qps, marker='s', markersize=8, linestyle='-', color='#2a9d8f', linewidth=2)
    axes[1].set_title('Throughput (QPS) vs Threads')
    axes[1].set_xlabel('Number of Threads')
    axes[1].set_ylabel('Queries Per Second (QPS)')
    axes[1].set_xticks(threads)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    for i, txt in enumerate(qps):
        axes[1].annotate(f"{txt:.0f}", (threads[i], qps[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # 图 2.3：平均延迟
    axes[2].plot(threads, latencies, marker='^', markersize=8, linestyle='-', color='#e76f51', linewidth=2)
    axes[2].set_title('Average Query Latency vs Threads')
    axes[2].set_xlabel('Number of Threads')
    axes[2].set_ylabel('Latency (us)')
    axes[2].set_xticks(threads)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    for i, txt in enumerate(latencies):
        axes[2].annotate(f"{txt:.1f}", (threads[i], latencies[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/scaling_performance.png')
    plt.close()

# ==========================================
# 3. Pareto 图 (Recall vs QPS)
# ==========================================
def plot_optimized_pareto():
    plt.figure(figsize=(10, 7), dpi=300)
    
    # 强制限定 Y 轴范围，使三个 HNSW 版本在视觉上处于同一水平线
    plt.ylim(0.985, 1.005)

    # 绘制辅助基准线，证明所有优化版本都在 0.995 附近
    plt.axhline(y=0.9950, color='gray', linestyle='--', alpha=0.5, zorder=1)
    plt.text(100, 0.9945, 'Recall Baseline (0.9950)', color='gray', fontsize=10, zorder=2)

    # 绘制散点
    for i in range(len(algorithms)):
        plt.scatter(qps_data[i], recall_data[i], label=algorithms[i], color=colors[i], s=150, edgecolor='black', zorder=3)

    # 优化文本重叠逻辑
    for i, txt in enumerate(algorithms):
        label_str = f"{recall_data[i]:.4f}"
        
        if txt == 'HNSW (2D)':
            # 向上偏移，错开线性数组的点
            offset = (0, 10) 
            ha_align = 'center'
            va_align = 'bottom'
        elif txt == 'HNSW (Linear)':
            # 向下偏移，错开 2D 版本的点
            offset = (0, -12) 
            ha_align = 'center'
            va_align = 'top'
        else:
            # 其他点正常放在右侧
            offset = (10, 0)
            ha_align = 'left'
            va_align = 'center'
            
        plt.annotate(label_str, 
                     (qps_data[i], recall_data[i]), 
                     textcoords="offset points", 
                     xytext=offset, 
                     ha=ha_align, 
                     va=va_align,
                     fontsize=11)

    plt.xscale('log')
    plt.title('Recall vs. Throughput (Pareto Front)', fontsize=14)
    plt.xlabel('Throughput (QPS) - Log Scale', fontsize=12)
    plt.ylabel('Recall Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.legend(loc='upper left', fontsize=11) 
    
    plt.tight_layout()
    plt.savefig('results/recall_vs_qps_optimized.png')
    plt.close()

if __name__ == "__main__":
    print("正在生成基础性能柱状图...")
    plot_base_bar_charts()
    
    print("正在生成多线程 Scaling 测试图表...")
    filepath = "results/HNSW_scaling_results.txt"
    plot_scaling_results(filepath)
    
    print("正在生成优化后的 Recall vs QPS 散点图...")
    plot_optimized_pareto()
    
    print("所有图表已成功生成并保存在 results/ 目录下！")