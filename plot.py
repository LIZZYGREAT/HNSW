import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 确保输出目录存在
output_dir = "./report/images"
os.makedirs(output_dir, exist_ok=True)

# 2. 核心实验数据 (从测试结果文本提取)
labels = ['FlatScan', 'HNSW (2D)', 'HNSW (Linear)', 'HNSW (Parallel)']

# 各项性能指标
qps = [93.98, 1234.57, 1176.47, 13333.33]
latency_us = [10640.97, 808.04, 849.57, 510.00]
build_time_ms = [0, 80261.00, 74091.00, 5582.00]  # FlatScan 无建图阶段
recall = [1.0000, 0.9965, 0.9965, 0.9955]

# 全局绘图样式设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
colors = ['#E63946', '#457B9D', '#1D3557', '#2A9D8F']

def add_value_labels(ax, spacing=5):
    """在柱状图顶部添加数值标签"""
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        if y_value > 0:
            ax.annotate(f'{y_value:,.0f}', (x_value, y_value), xytext=(0, spacing),
                        textcoords="offset points", ha='center', va='bottom', fontsize=10)

# --- 图 1：吞吐量对比 (QPS) ---
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, qps, color=colors, edgecolor='black', zorder=3)
ax.set_title('Throughput Comparison (QPS)')
ax.set_ylabel('Queries Per Second')
ax.set_yscale('log') # 使用对数坐标轴，直观展现数量级的飞跃
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
add_value_labels(ax)
plt.tight_layout()
plt.savefig(f"{output_dir}/throughput_qps.pdf", format='pdf', dpi=300)
plt.savefig(f"{output_dir}/throughput_qps.png", format='png', dpi=300)
plt.close()

# --- 图 2：平均查询延迟 (Latency) ---
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, latency_us, color=colors, edgecolor='black', zorder=3)
ax.set_title('Average Query Latency (us)')
ax.set_ylabel('Latency (microseconds)')
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
add_value_labels(ax)
plt.tight_layout()
plt.savefig(f"{output_dir}/query_latency.pdf", format='pdf', dpi=300)
plt.savefig(f"{output_dir}/query_latency.png", format='png', dpi=300)
plt.close()

# --- 图 3：建图耗时对比 (Build Time) ---
# 排除 FlatScan，对比剩余三项
build_labels = labels[1:]
build_times = build_time_ms[1:]
fig, ax = plt.subplots(figsize=(7, 6))
bars = ax.bar(build_labels, build_times, color=colors[1:], edgecolor='black', zorder=3, width=0.6)
ax.set_title('Index Build Time Comparison (ms)')
ax.set_ylabel('Time (milliseconds)')
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
add_value_labels(ax)
plt.tight_layout()
plt.savefig(f"{output_dir}/build_time.pdf", format='pdf', dpi=300)
plt.savefig(f"{output_dir}/build_time.png", format='png', dpi=300)
plt.close()

# --- 图 4：召回率与吞吐量的权衡 (Recall vs QPS Pareto Front) ---
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(labels)):
    ax.scatter(qps[i], recall[i], color=colors[i], s=150, edgecolor='black', zorder=3, label=labels[i])

ax.set_title('Recall vs. Throughput (Pareto Front)')
ax.set_xlabel('Throughput (QPS) - Log Scale')
ax.set_ylabel('Recall Rate')
ax.set_xscale('log')
ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
ax.legend(loc='lower left')

# 为散点添加文字标注
for i, txt in enumerate(labels):
    ax.annotate(f" {recall[i]:.4f}", (qps[i], recall[i]), xytext=(5, 0), textcoords='offset points', va='center')

plt.tight_layout()
plt.savefig(f"{output_dir}/recall_vs_qps.pdf", format='pdf', dpi=300)
plt.savefig(f"{output_dir}/recall_vs_qps.png", format='png', dpi=300)
plt.close()

print(f"[SUCCESS] All 4 evaluation charts generated and saved to '{output_dir}/'")