from vllm import LLM, SamplingParams
import json
def get_alpaca(dataset_path):
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["instruction"], data["output"])
        for data in dataset
    ]
    return dataset

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.95, top_p=1, max_tokens=2048)

# Create an LLM.
llm = LLM(model="/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
datasets = get_alpaca("/workspace/alpaca_data.json")
for data in datasets:
    print(data["instruction"])
    outputs = llm.generate(data["instruction"], sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
