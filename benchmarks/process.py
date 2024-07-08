file_path = "/home/jovyan/hhy/vllm-hhy/benchmarks/logs/log_"
lengths = [1024, 2048, 4080]
ratios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

outputs = []
for length in lengths:
    slots = []
    for ratio in ratios:
        name = file_path + str(length) + "_" + str(ratio) + ".txt"
        nums = []
        with open(name, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if i >= 4:
                    data = float(line.strip().split(' ')[-1])
                    nums.append(data)
        slots.append(sum(nums) / len(nums))
    outputs.append(slots)
print(outputs)