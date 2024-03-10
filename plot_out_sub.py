import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd
import glob
import os 
import matplotlib.pyplot as plt

d_all = {"1_rate":[], "2_rate":[], "3_rate":[], "4_rate":[], "input":["A", "B", "C", "D", "E"]}
total_count = 42 * 2

df_list = []
for file in glob.glob(os.path.join("/mnt/sda/jinyu/attack_vc/subjective", "*.csv")):
    if "white" in file:
        df = pd.read_csv(file)
        df_list.append(df)  
        
total_count = 4*42
for input in ["1.wav", "0.wav", "2.wav"]:
    count_d = {"1" : 0, "2": 0, "3": 0, "4": 0}
    df_ = pd.concat([df[input] for df in df_list], axis = 1)
    for index,row in df_.iterrows():
        L = row.values.flatten().tolist()
        L.sort()
        L = L[1:-1]
        for v in L:
            count_d[str(v)] += 1
    for key in count_d.keys():
        d_all[f"{key}_rate"].append(count_d[key] / total_count * 100)

df_list = []
for file in glob.glob(os.path.join("/mnt/sda/jinyu/attack_vc/subjective", "*.csv")):
    if "black" in file:
        df = pd.read_csv(file)
        df_list.append(df)
    

for input in ["0.wav", "2.wav"]:
    count_d = {"1" : 0, "2":0, "3": 0, "4": 0}
    df_ = pd.concat([df[input] for df in df_list], axis = 1)
    for index,row in df_.iterrows():
        L = row.values.flatten().tolist()
        L.sort()
        L = L[1:-1]
        for v in L:
            count_d[str(v)] += 1
    for key in count_d.keys():
        d_all[f"{key}_rate"].append(count_d[key] / total_count * 100)
d_all = pd.DataFrame(d_all)
# fig = plt.figure()
# ax = plt.subplot(111)
# bar_width = 1.0
# plt.rcParams["font.family"] = "serif"
# ax.bar(list(range(1,22,5)), d_all["1_rate"], width=bar_width, label="Different, absolutely sure")
# ax.bar(list(range(2,23,5)), d_all["2_rate"], width=bar_width, label="Different, but not very sure")
# ax.bar(list(range(3,24,5)), d_all["3_rate"], width=bar_width, label="Same, but not very sure")
# ax.bar(list(range(4,25,5)), d_all["4_rate"], width=bar_width, label="Same, absolutely sure")

# plt.xticks([2.5, 7.5, 12.5, 17.5, 22.5], ["adv. \ninput", "adv. \noutput", "adv. \ninput", "adv. \noutput", "ori. \noutput"])

# ax2 = ax.twiny()
# ax2.set_xticks(ax.get_xticks())
# ax2.set_xbound(ax.get_xbound())
# ax2.set_xticklabels(["(a)", "(b)", "(c)", "(d)", "(e)"])
# plt.tick_params(axis='x', width = 0)



# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
#           fancybox=True, shadow=True, ncol=2)
# plt.savefig("plot_sub.jpg", bbox_inches="tight")
x = [1,3,4,6,7]
fig = plt.figure(dpi=150)
ax = plt.subplot(111)
bar_width = 0.9
plt.rcParams["font.family"] = "serif"
ax.bar(x, d_all["1_rate"], width=bar_width, label="Different, absolutely sure", color = "#4091c9")
ax.bar(x, d_all["2_rate"], width=bar_width, bottom=d_all["1_rate"], label="Different, but not very sure", color = "#9dcee2")
ax.bar(x, d_all["3_rate"], width=bar_width, bottom=d_all["1_rate"] + d_all["2_rate"], label="Same, but not very sure", color = "#ffccd5")
ax.bar(x, d_all["4_rate"], width=bar_width, bottom=d_all["1_rate"] + d_all["2_rate"] + d_all["3_rate"], label="Same, absolutely sure", color = "#ff758f")

for c in ax.containers:

    # Optional: if the segment is small or 0, customize the labels
    labels = [round(v.get_height(),2) if v.get_height() > 5 else '' for v in c]
    
    # remove the labels parameter if it's not needed for customized labels
    ax.bar_label(c, labels=labels, label_type='center')
plt.xticks(x, ["ori. \noutput", "adv. \ninput", "adv. \noutput", "adv. \ninput", "adv. \noutput"])
plt.ylabel("Percentage(%)")

ax2 = ax.twiny()
ax2.set_xticks(ax.get_xticks())
ax2.set_xbound(ax.get_xbound())
ax2.set_xticklabels(["(a)", "(b)", "(c)", "(d)", "(e)"])
plt.tick_params(axis='x', width = 0)



ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2)
plt.savefig("plot_sub.jpg", bbox_inches="tight")
