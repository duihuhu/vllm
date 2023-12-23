import numpy as np
root_name = "logs_6_"
sep_name = [1, 2, 3, 4, 5]
mean = []
a = []
for sep in sep_name:
    file_name = root_name + str(sep) + ".txt"
    with open(file_name, 'r') as file:
        datas = []
        lines = file.readlines()
        for i, line in enumerate(lines):
            data = line.strip().split(',')
            st = float(data[-2].split(' ')[-1])
            ed = float(data[-1].split(' ')[-1])
            if i == 0:
                continue
            else:
                time_slot = ed - st
                datas.append(time_slot)
        a.append(datas)
        datas = np.array(datas)
        mean.append(np.mean(datas))
        print(datas)
        print(mean)
mean = np.array(mean)
print(np.mean(mean))
