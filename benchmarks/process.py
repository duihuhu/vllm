root_path = "/home/jovyan/vllm/benchmarks/logs/"
dir_path = "log_bd/"
prefix_path = "bd_"
suffix_path = ".txt"
bs = [1,2,3]
total_len = [8,16,32,64,128,256,512,1024,2048,4096]

for lens in total_len:
    ffn1 = []
    ffn2 = []
    qkv_proj = []
    rope = []
    attn = []
    o_proj = []
    layer = []
    for b in bs:
        file_name = root_path + dir_path + prefix_path + str(lens) + "_" + str(b) + suffix_path
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = line.strip().split(' ')
                if str(data[0]) == "ffn1":
                    ffn1.append(float(data[-2]))
                elif str(data[0]) == "ffn2":
                    ffn2.append(float(data[-2]))
                elif str(data[0]) == "qkv_proj":
                    qkv_proj.append(float(data[-2]))
                elif str(data[0]) == "rope":
                    rope.append(float(data[-2]))
                elif str(data[0]) == "attn":
                    attn.append(float(data[-2]))
                elif str(data[0]) == "o_proj":
                    o_proj.append(float(data[-2]))
                elif str(data[0]) == "layer":
                    layer.append(float(data[-2]))
    ffn1_m = sum(ffn1) / len(ffn1)
    ffn2_m = sum(ffn2) / len(ffn2)
    qkv_proj_m = sum(qkv_proj) / len(qkv_proj)
    rope_m = sum(rope) / len(rope)
    attn_m = sum(attn) / len(attn)
    o_proj_m = sum(o_proj) / len(o_proj)
    layer_m = sum(layer) / len(layer)
    print(f"----------{lens}-----------")
    print(f"ffn1 ffn2 qkv_proj rope attn o_proj others")
    print(f"{ffn1_m / layer_m:.4f}")
    print(f"{ffn2_m / layer_m:.4f}")
    print(f"{qkv_proj_m / layer_m:.4f}")
    print(f"{rope_m / layer_m:.4f}")
    print(f"{attn_m / layer_m:.4f}")
    print(f"{o_proj_m / layer_m:.4f}")
    print(f"{(layer_m - ffn1_m - ffn1_m - qkv_proj_m - rope_m - attn_m - o_proj_m) / layer_m:.4f}")