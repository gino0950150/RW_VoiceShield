import random
with open("/homes/jinyu/attack_vc/data/test_pairs.txt", "r") as f:
    lines = f.readlines()
lines = [line.rstrip().split() for line in lines] 
r = list(range(len(lines)))
random.shuffle(r)
dictionary_count = {}
selected_lines = []
for j in r:
    src_path, tar_path, adv_path = lines[j]
    s_name = tar_path.split("/")[-2] 
    if s_name not in dictionary_count.keys():
        dictionary_count[s_name] = 0
    if dictionary_count[s_name] >= 2:
        continue
    else:
        selected_lines.append([src_path, tar_path, adv_path])
        dictionary_count[s_name] += 1
print(dictionary_count)
input()
with open("/homes/jinyu/attack_vc/data/sub_test_pairs.txt", "a") as f:
    for src_path, tar_path, adv_path in selected_lines:
        f.writelines([f"{src_path} {tar_path} {adv_path}\n"])