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
    
df = pd.concat(df_list, axis=0)    
print(df)    
        

for input in ["0.wav", "2.wav"]:
    count_d = {"1" : 0, "2":0, "3": 0, "4": 0}
    values = df[input].value_counts().keys().tolist()
    counts = df[input].value_counts().tolist()
    
    for val , ct in zip(values, counts):
        count_d[str(val)] += ct
    for key in count_d.keys():
        d_all[f"{key}_rate"].append(count_d[key] / total_count * 100)

df_list = []
for file in glob.glob(os.path.join("/mnt/sda/jinyu/attack_vc/subjective", "*.csv")):
    if "black" in file:
        df = pd.read_csv(file)
        df_list.append(df)
    
df = pd.concat(df_list, axis=0)    
print(df)    
        
for input in ["0.wav", "2.wav", "1.wav"]:
    count_d = {"1" : 0, "2":0, "3": 0, "4": 0}
    values = df[input].value_counts().keys().tolist()
    counts = df[input].value_counts().tolist()
    
    for val , ct in zip(values, counts):
        count_d[str(val)] += ct
    for key in count_d.keys():
        d_all[f"{key}_rate"].append(count_d[key] / total_count * 100)
d_all = pd.DataFrame(d_all)
print(d_all)
x = [1,2,4,5,7]

fig = plt.figure()
ax = plt.subplot(111)
bar_width = 1.0
plt.rcParams["font.family"] = "serif"
ax.bar(x, d_all["1_rate"], width=bar_width, label="Different, absolutely sure", color = "#023047")
ax.bar(x, d_all["2_rate"], width=bar_width, bottom=d_all["1_rate"], label="Different, but not very sure")
ax.bar(x, d_all["3_rate"], width=bar_width, bottom=d_all["1_rate"] + d_all["2_rate"], label="Same, but not very sure")
ax.bar(x, d_all["4_rate"], width=bar_width, bottom=d_all["1_rate"] + d_all["2_rate"] + d_all["3_rate"], label="Same, absolutely sure")

for c in ax.containers:

    # Optional: if the segment is small or 0, customize the labels
    labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
    
    # remove the labels parameter if it's not needed for customized labels
    ax.bar_label(c, labels=labels, label_type='center')
    
plt.xticks(x, ["adv. \ninput", "adv. \noutput", "adv. \ninput", "adv. \noutput", "ori. \noutput"])
plt.ylabel("Percentage(%)")

ax2 = ax.twiny()
ax2.set_xticks(ax.get_xticks())
ax2.set_xbound(ax.get_xbound())
ax2.set_xticklabels(["(a)", "(b)", "(c)", "(d)", "(e)"])
plt.tick_params(axis='x', width = 0)



ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2)
plt.savefig("plot_sub.jpg", bbox_inches="tight")
