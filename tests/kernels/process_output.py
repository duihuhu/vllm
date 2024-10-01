import pandas as pd
import os

def get_input(data, i):
    a = data[i]
    #b = data[i+1]
    c = data[i+2]
    '''d = data[i+3]
    x = 0
    y = 0
    z = 0
    if b != 0:
        x = a / b
    else:
        x = 0
    if d != 0:
        y = c / d
    else:
        y = 0
    z = x + y
    return z'''
    return a + c

file_path_1 = "/home/jovyan/vllm/tests/kernels/profile_logs_long_range/long_range_1_"
file_path_2 = ".ncu-rep"
file_path_3 = ".csv"
lengths = []
i = 2048
while True:
    if i > 4096:
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

st = 975
ed = 1374
layers = 40
iter_length = 10

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
    act2= []
    ffn22 = []

    file_path = file_path_1 + str(length) + file_path_3
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        id = int(str(row['ID']))
        mname = str(row['Metric Name'])
        if id >= st and id <= ed and mname == "gpu__time_duration.avg":
            data1 = int(str(row['Grid Size']).split("(")[1].split(",")[0])
            data2 = float(str(row['Metric Value']))

            offset = (id - st) % iter_length
            if offset == 0:
                preattnnorm.append(data1)
                preattnnorm2.append(data2)
            elif offset == 1:
                qkvproj.append(data1)
                qkvproj2.append(data2)
            elif offset == 2:
                rope.append(data1)
                rope2.append(data2)
            elif offset == 3:
                store.append(data1)
                store2.append(data2)
            elif offset == 4:
                attn.append(data1)
                attn2.append(data2)
            elif offset == 5:
                oproj.append(data1)
                oproj2.append(data2)
            elif offset == 6:
                postattnnorm.append(data1)
                postattnnorm2.append(data2)
            elif offset == 7:
                ffn1.append(data1)
                ffn12.append(data2)
            elif offset == 8:
                act.append(data1)
                act2.append(data2)
            else:
                ffn2.append(data1)
                ffn22.append(data2)

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

print("----------BLOCK SIZE----------")
print("\n")
for i, name in enumerate(names):
    print(f"----------{name}----------")
    for item in outputs1[i]:
        print(item)

print("\n")

print("----------Time----------")
print("\n")
for i, name in enumerate(names):
    print(f"----------{name}----------")
    for item in outputs2[i]:
        print(item)


'''iter_length = 10
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
            data = float(row['Metric Value'])
            
            exp = 1
            if row['Metric Unit'] == "byte" or row['Metric Unit'] == "byte/second":
                exp = 1
            elif row['Metric Unit'] == "Kbyte" or row['Metric Unit'] == "Kbyte/second":
                exp = 1024
            elif row['Metric Unit'] == "Mbyte" or row['Metric Unit'] == "Mbyte/second": 
                exp = 1024 * 1024
            elif row['Metric Unit'] == "Gbyte" or row['Metric Unit'] == "Gbyte/second":
                exp = 1024 * 1024 * 1024
            else: 
                exp = 1024 * 1024 * 1024 * 1024

            data = data * exp         
            if offset == 0:
                preattnnorm.append(data)
                if row['Metric Unit'] == "sector":
                    qkvproj.append(int(data))
                else:
                    qkvproj2.append(float(data))
            elif offset == 1:
                qkvproj.append(data)
                if row['Metric Unit'] == "sector":
                    qkvproj.append(int(data))
                else:
                    qkvproj2.append(float(data))
            elif offset == 2:
                rope.append(data)
                if row['Metric Unit'] == "sector":
                    rope.append(int(data))
                else:
                    rope2.append(float(data))
            elif offset == 3:
                store.append(data)
                if row['Metric Unit'] == "sector":
                    store.append(int(data))
                else:
                    store2.append(float(data))
            elif offset == 4:
                attn.append(data)
                if row['Metric Unit'] == "sector":
                    attn.append(int(data))
                else:
                    attn2.append(float(data))
            elif offset == 5:
                oproj.append(data)
                if row['Metric Unit'] == "sector":
                    oproj.append(int(data))
                else:
                    oproj2.append(float(data))
            elif offset == 6:
                postattnnorm.append(data)
                if row['Metric Unit'] == "sector":
                    postattnnorm.append(int(data))
                else:
                    postattnnorm2.append(float(data))
            elif offset == 7:
                ffn1.append(data)
                if row['Metric Unit'] == "sector":
                    ffn1.append(int(data))
                else:
                    ffn12.append(float(data))
            elif offset == 8:
                act.append(data)
                if row['Metric Unit'] == "sector":
                    act.append(int(data))
                else:
                    act2.append(float(data))
            else:
                ffn2.append(data)
                if row['Metric Unit'] == "sector":
                    ffn2.append(int(data))
                else:
                    ffn22.append(float(data))
    
    t = 0
    total_length = layers * 4
    while t < total_length:
        d1 = get_input(preattnnorm, t)
        preattnnorm2.append(d1)
        d2 = get_input(qkvproj, t)
        qkvproj2.append(d2)
        d3 = get_input(rope, t)
        rope2.append(d3)
        d4 = get_input(store, t)
        store2.append(d4)
        d5 = get_input(attn, t)
        attn2.append(d5)
        d6 = get_input(oproj, t)
        oproj2.append(d6)
        d7 = get_input(postattnnorm, t)
        postattnnorm2.append(d7)
        d8 = get_input(ffn1, t)
        ffn12.append(d8)
        d9 = get_input(act, t)
        act2.append(d9)
        d10 = get_input(ffn2, t)
        ffn22.append(d10)
        t += 4


    total_preattnnorm.append(sum(preattnnorm2) / layers)
    total_qkvproj.append(sum(qkvproj2) / layers)
    total_rope.append(sum(rope2) / layers)
    total_store.append(sum(store2) / layers)
    total_attn.append(sum(attn2) / layers)
    total_oproj.append(sum(oproj2) / layers)
    total_postattnnorm.append(sum(postattnnorm2) / layers)
    total_ffn1.append(sum(ffn12) / layers)
    total_act.append(sum(act2) / layers)
    total_ffn2.append(sum(ffn22) / layers)

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
#outputs2 = []

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

#print(f"----------TRANSACTIONS----------")
print(f"----------BYTES----------")

for j, name in enumerate(names):
    print(f"----------{name}----------")
    #print(len(outputs1[i]))
    for data2 in outputs1[j]:
        print(data2)
    print(f"----------END----------")

print(f"----------UTILIZATION----------")

for i, name in enumerate(names):
    print(f"----------{name}----------")
    for data in outputs2[i]:
        print(data)
    print(f"----------END----------")'''