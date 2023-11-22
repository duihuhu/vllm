from vllm import LLM, SamplingParams
import json
def get_alpaca(dataset_path):
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Only keep the first two turns of each conversation.

    alpaca_data = []
    for data in dataset:
        if data['input'] == "":
            alpaca_data.append((data["instruction"], data["output"]))

    return alpaca_data

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.95, top_p=1, max_tokens=2048)
datasets = get_alpaca("/workspace/alpaca_data.json")

# Create an LLM.
llm = LLM(model="/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
opt_answer_datasets = []
for data in datasets:
    js = {}
    outputs = llm.generate(data[0], sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        js['instruction'] = data[0]
        js['output_text'] = generated_text
        js['input'] = ""
        js['output'] = len(output.outputs[0].token_ids)
        js['stop_reason'] = output.outputs[0].finish_reason
        opt_answer_datasets.append(js)
        
with open("alpaca_opt13b_answer.json", 'w+') as json_file:
    json.dump(opt_answer_datasets, json_file, indent=2) 