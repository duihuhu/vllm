import pandas as pd

file_path = "/home/jovyan/vllm/tests/kernels/temp.csv"
df = pd.read_csv(file_path, header = 0)
kernels = set()
datas = {}

for row in df:
    if row['Kernel Name'] in kernels:
        continue
    else:
        kernels.add(row['Kernel Name'])
        print(f"{row['Block Size', 'Grid Size']}\n")