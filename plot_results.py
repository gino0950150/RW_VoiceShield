import matplotlib.pyplot as plt
result_path = "/mnt/sda/jinyu/attack_vc/exps/0217_blackbox512/ephs_results_ecapa.txt"
with open(result_path, "r") as f:
    lines = f.readlines()

iters = []
delta = []
attak_success_rate = []
preserve_success_rate = []
pesq = []

for line in lines:
    line = line.rstrip().split()
    iters.append(int(line[1]))
    delta.append(float(line[3]))
    attak_success_rate.append(float(line[5]))
    preserve_success_rate.append(float(line[7]))
    pesq.append(float(line[-3]))
    
# 创建主要的y轴
fig, ax1 = plt.subplots()
fig.set_figwidth(12)
# 绘制第一个数据集
l1 = ax1.plot(delta, attak_success_rate, 'r-o', label='ASR')
l2 = ax1.plot(delta, preserve_success_rate, 'b-o', label='PSR')
ax1.set_xlabel('SNR(dB)')
ax1.set_ylabel('Percentage')

# 创建第二个y轴，并共享x轴
ax2 = ax1.twinx()
ax2.set_ylabel('Percentage')
l3 = ax2.plot(delta, pesq,"c-o", label='PESQ')
ax2.set_ylabel('PESQ')

lns = l1+l2+l3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="right")


# 添加图例
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='lower right')

# 显示图表
plt.savefig("plot_0.1.jpg")