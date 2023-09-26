from vllm import LLM, SamplingParams

# Sample prompts.
debug_prompts = [
    "The capital of France is",
]

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(use_beam_search = True, top_p=0.95)

# Create an LLM.
llm = LLM(model="/workspace/models/facebook/opt-125m", tensor_parallel_size = 2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(debug_prompts, sampling_params, use_tqdm = False)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
