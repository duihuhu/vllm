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
        else: 
            alpaca_data.append((data["instruction"] + data['input'], data["output"]))

    return dataset

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.95, top_p=1, max_tokens=2048)
datasets = get_alpaca("/workspace/alpaca_data_test.json")

# Create an LLM.
llm = LLM(model="/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
for data in datasets:
    outputs = llm.generate(data[0], sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
