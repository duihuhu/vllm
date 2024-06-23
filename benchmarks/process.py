root_path = "/home/jovyan/hhy/vllm-hhy/benchmarks/"
dir_path = "logs2/"
prefix_path = "log2_"
suffix_path = ".txt"
total_len = [32,64,128,256,512,1024,2048,4080]
ratio = [10,20,30,40,50,60,70,80,90,100]

total_costs = []
for tl in total_len:
    tl_costs = []
    for r in ratio:
        file_name = root_path + dir_path  + prefix_path + str(tl) + "_" + str(r) + suffix_path
        costs = []
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for i, in range(len(lines)):
                if i == 8 or i == 10 or i == 12:
                    st = float(lines[i].strip().split(' ')[-1])
                    ed = float(lines[i+1].strip().split(' ')[-1])
                    costs.append(ed - st)
        tl_costs.append(sum(costs) / len(costs))
    total_costs.append(tl_costs)
print(total_costs)    