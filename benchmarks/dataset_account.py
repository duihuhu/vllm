import json
from transformers import AutoTokenizer
import numpy as np

def process(prompts_lens):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    h = 0
    i = 0

    for prompt in prompts_lens:
        if prompt <= 8:
            a += 1
        elif prompt > 8 and prompt <= 16:
            b += 1
        elif prompt > 16 and prompt <= 32:
            c += 1
        elif prompt > 32 and prompt <= 64:
            d += 1
        elif prompt > 64 and prompt <= 128:
            e += 1
        elif prompt > 128 and prompt <= 256:
            f += 1
        elif prompt > 256 and prompt <= 512:
            g += 1
        elif prompt > 512 and prompt <= 1024:
            h += 1
        else:
            i += 1

    print(f"{a / len(prompts_lens):.2f}")
    print(f"{b / len(prompts_lens):.2f}")
    print(f"{c / len(prompts_lens):.2f}")
    print(f"{d / len(prompts_lens):.2f}")
    print(f"{e / len(prompts_lens):.2f}")
    print(f"{f / len(prompts_lens):.2f}")
    print(f"{g / len(prompts_lens):.2f}")
    print(f"{h / len(prompts_lens):.2f}")
    print(f"{i / len(prompts_lens):.2f}")

tokenizer_path = "/workspace/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

dataset_path = "/workspace/ShareGPT_V3_unfiltered_cleaned_split.json"
with open(dataset_path) as f:
    dataset = json.load(f)
dataset = [data for data in dataset if len(data["conversations"]) >= 2]
chat_dataset = []
for data in dataset:
    data_len = len(data["conversations"])
    i = 0
    while i < data_len:
        j = i + 1
        if j < data_len:
            chat_dataset.append((data["conversations"][i]["value"], data["conversations"][j]["value"]))
            i = j + 1
        else:
            break
prompts = [prompt for prompt, _ in chat_dataset]
prompt_token_ids = tokenizer(prompts).input_ids
outputs = [output for _, output in chat_dataset]
ouput_token_ids = tokenizer(outputs).input_ids
tokenizer_dataset = []
for i in range(len(chat_dataset)):
    tokenizer_dataset.append((len(prompt_token_ids[i]), len(ouput_token_ids[i])))
filtered_dataset = []
for data in tokenizer_dataset:
    if data[0] > 2048 or data[1] > 2048 or data[0] + data[1] > 2048:
        continue
    filtered_dataset.append(data)

file_path = "/workspace/vllm/benchmarks/account.txt"
with open(file_path, 'a') as file:
    for data in filtered_dataset:
        file.write(f"prompt tokens len {data[0]}, output tokens len {data[1]}/n")

prompts_lens = [prompt for prompt, _ in filtered_dataset]
outputs_lens = [output for _, output in filtered_dataset]
prompts_lens = np.array(prompts_lens)
outputs_lens = np.array(outputs_lens)

print(np.median(prompts_lens))
print(np.mean(prompts_lens))
print(np.median(outputs_lens))
print(np.mean(outputs_lens))

process(prompts_lens)
process(outputs_lens)