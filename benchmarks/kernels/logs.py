input_len = [64, 128, 256, 512, 1024]
output_len = [64, 128, 256, 512, 1024]

agg = [0, 1]

agg_ttft = []
agg_tbt = []

non_agg_ttft = []
non_agg_tbt = []

for is_agg in agg:
    for il in input_len:
        ttft = []
        tbt = []
        for ol in output_len:
            file_path = "/home/jovyan/hhy/vllm-hhy/benchmarks/logs/" + "log_" + str(is_agg) + "_" + str(il) + "_" + str(ol) + ".txt"
            a_ttft = []
            a_tbt = []
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    data = line.strip().split(' ')
                    ite = int(data[1])
                    tm = float(data[-1])
                    if ite == 0:
                        a_ttft.append(tm)
                    else:
                        a_tbt.append(tm)
            ttft.append(sum(a_ttft) / len(a_ttft))
            tbt.append(sum(a_tbt) / len(a_tbt))
    if is_agg == 0:
        non_agg_ttft.append(ttft)
        non_agg_tbt.append(tbt)
    else:
        agg_ttft.append(ttft)
        agg_tbt.append(tbt)

print(agg_ttft)
print(agg_tbt)
print(non_agg_ttft)
print(non_agg_tbt)
                            