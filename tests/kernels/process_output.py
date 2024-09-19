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

'''for length in lengths:
    file_path_in = file_path_1 + str(length) + file_path_2
    file_path_out = file_path_1 + str(length) + file_path_3
    base_command = "ncu -i {x} --page details --csv --log-file {y}"
    command = base_command.format(x = file_path_in, y = file_path_out)
    os.system(command)'''

iter_length = 10
st = 975
ed = 1374
layers = 40

total_preattnnorm = []
total_qkvproj = []
total_rope = []
total_store = []
total_attn = []
total_oproj = []
total_postattnnorm = []
total_ffn1 = []
total_act = []
total_ffn2 = []

total_preattnnorm2 = []
total_qkvproj2 = []
total_rope2 = []
total_store2 = []
total_attn2 = []
total_oproj2 = []
total_postattnnorm2 = []
total_ffn12 = []
total_act2 = []
total_ffn22 = []

for length in lengths:
    preattnnorm = []
    qkvproj =  []
    rope = []
    store = []
    attn = []
    oproj = []
    postattnnorm = []
    ffn1 = []
    act = []
    ffn2 = []

    preattnnorm2 = []
    qkvproj2 =  []
    rope2 = []
    store2 = []
    attn2 = []
    oproj2 = []
    postattnnorm2 = []
    ffn12 = []
    act2 = []
    ffn22 = []
    
    file_path = file_path_1 + str(length) + file_path_3
    df = pd.read_csv(file_path)
    
    for _, row in df.iterrows():
        id = int(row['ID'])
        if id >= st and id <= ed:
            offset = (id - st) % iter_length
            data = str(row['Metric Value']).replace(",", "")
            if offset == 0:
                if row['Metric Unit'] == "sector":
                    preattnnorm.append(int(data))
                else:
                    preattnnorm2.append(float(data))
            elif offset == 1:
                if row['Metric Unit'] == "sector":
                    qkvproj.append(int(data))
                else:
                    qkvproj2.append(float(data))
            elif offset == 2:
                if row['Metric Unit'] == "sector":
                    rope.append(int(data))
                else:
                    rope2.append(float(data))
            elif offset == 3:
                if row['Metric Unit'] == "sector":
                    store.append(int(data))
                else:
                    store2.append(float(data))
            elif offset == 4:
                if row['Metric Unit'] == "sector":
                    attn.append(int(data))
                else:
                    attn2.append(float(data))
            elif offset == 5:
                if row['Metric Unit'] == "sector":
                    oproj.append(int(data))
                else:
                    oproj2.append(float(data))
            elif offset == 6:
                if row['Metric Unit'] == "sector":
                    postattnnorm.append(int(data))
                else:
                    postattnnorm2.append(float(data))
            elif offset == 7:
                if row['Metric Unit'] == "sector":
                    ffn1.append(int(data))
                else:
                    ffn12.append(float(data))
            elif offset == 8:
                if row['Metric Unit'] == "sector":
                    act.append(int(data))
                else:
                    act2.append(float(data))
            else:
                if row['Metric Unit'] == "sector":
                    ffn2.append(int(data))
                else:
                    ffn22.append(float(data))
    
    total_preattnnorm.append(sum(preattnnorm) / layers)
    total_qkvproj.append(sum(qkvproj) / layers)
    total_rope.append(sum(rope) / layers)
    total_store.append(sum(store) / layers)
    total_attn.append(sum(attn) / layers)
    total_oproj.append(sum(oproj) / layers)
    total_postattnnorm.append(sum(postattnnorm) / layers)
    total_ffn1.append(sum(ffn1) / layers)
    total_act.append(sum(act) / layers)
    total_ffn2.append(sum(ffn2) / layers)

    total_preattnnorm2.append(sum(preattnnorm2) / layers)
    total_qkvproj2.append(sum(qkvproj2) / layers)
    total_rope2.append(sum(rope2) / layers)
    total_store2.append(sum(store2) / layers)
    total_attn2.append(sum(attn2) / layers)
    total_oproj2.append(sum(oproj2) / layers)
    total_postattnnorm2.append(sum(postattnnorm2) / layers)
    total_ffn12.append(sum(ffn12) / layers)
    total_act2.append(sum(act2) / layers)
    total_ffn22.append(sum(ffn22) / layers)

outputs1 = []
outputs2 = []

outputs1.append(total_preattnnorm)
outputs1.append(total_qkvproj)
outputs1.append(total_rope)
outputs1.append(total_store)
outputs1.append(total_attn)
outputs1.append(total_oproj)
outputs1.append(total_postattnnorm)
outputs1.append(total_ffn1)
outputs1.append(total_act)
outputs1.append(total_ffn2)
    
outputs2.append(total_preattnnorm2)
outputs2.append(total_qkvproj2)
outputs2.append(total_rope2)
outputs2.append(total_store2)
outputs2.append(total_attn2)
outputs2.append(total_oproj2)
outputs2.append(total_postattnnorm2)
outputs2.append(total_ffn12)
outputs2.append(total_act2)
outputs2.append(total_ffn22)

names= ["preattnnorm", "qkvproj", "rope", "store", "attn", "oproj", "postattnnorm", "ffn1", "act", "ffn2"]

print(f"----------TRANSACTIONS----------")

for i, name in enumerate(names):
    print(f"----------{name}----------")
    for data in outputs1[i]:
        print(data)
    print(f"----------END----------")

print(f"----------UTILIZATION----------")

for i, name in enumerate(names):
    print(f"----------{name}----------")
    for data in outputs2[i]:
        print(data)
    print(f"----------END----------")