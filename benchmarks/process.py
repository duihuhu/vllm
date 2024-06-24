root_path = "/home/jovyan/hhy/vllm-hhy/benchmarks/"
dir_path = "logs6/"
prefix_path = "log_"
suffix_path = ".txt"
bs = [32,64]
total_len = [1024,2048,4080]
ratio = [10,20,30,40,50,60,70,80,90,100]

for b in bs:
    total_costs = []
    for tl in total_len:
        tl_costs = []
        '''file_name = root_path + dir_path  + prefix_path + str(tl) + suffix_path
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if i >= 3:
                    data = float(lines[i].strip().split(' ')[-1])
                    tl_costs.append(data)'''
        for r in ratio:
            file_name = root_path + dir_path  + prefix_path + str(b) + "_" + str(tl) + '_' + str(r) + suffix_path
            costs = []
            with open(file_name, 'r') as file:
                lines = file.readlines()
                length = len(lines)
                for i in range(length):
                    if i >= 4: #i == (length - 1) or i == (length - 9) or i == (length - 17):
                        #st = float(lines[i - 4].strip().split(' ')[-1])
                        #ed = float(lines[i].strip().split(' ')[-1])
                        data = float(lines[i].strip().split(' ')[-1])
                        #costs.append(ed - st)
                        costs.append(data)
            tl_costs.append(sum(costs) / len(costs))
            #tl_costs.append(float(lines[0].strip().split(' ')[-1]))
        total_costs.append(tl_costs)
        #total_costs.append(sum(tl_costs) / len(tl_costs))
    print(b)
    print("--------------------")
    print(total_costs)    
    print("--------------------")