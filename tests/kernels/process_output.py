import pandas as pd
import os

file_path_1 = "/home/jovyan/vllm/tests/kernels/profile_logs_transaction/total_transaction_1_"
file_path_2 = ".ncu-rep"
file_path_3 = ".csv"
lengths = [8,16,32]
i = 64
while True:
    if i > 2048:
        break
    else:
        lengths.append(i)
        i += 64

for length in lengths:
    file_path_in = file_path_1 + str(length) + file_path_2
    file_path_out = file_path_1 + str(length) + file_path_3
    base_command = "ncu -i {x} --page details --csv --log-file {y}"
    command = base_command.format(x = file_path_in, y = file_path_out)
    os.system(command)

for length in lengths:
    file_path = file_path_1 + str(length) + file_path_3
    df = pd.read_csv(file_path)
    for index, row in df[::-1].iterrows():
        if int(row['ID']) >= 1305 and int(row['ID']) <= 1314:
            print(f"Function Name: {row['Function Name']}")
            print(f"Grid Size: {row['Grid Size']}")
            print(f"Block Size: {row['Block Size']}")