import json
def get_alpaca(dataset_path):
    with open(dataset_path) as f:
        dataset = json.load(f)
        
    alpaca_data = []
    for data in dataset:
      js = {}
      if data['stop_reason'] == "stop":
        js['instruction'] = data['instruction']
        js['input'] = data['input']
        js['output'] = data['output']
        alpaca_data.append(js)
    return alpaca_data

def union_dataset(filename1, filename2):
  data = []
  with open(filename1) as f:
    dataset = json.load(f)
  for d in dataset:
    data.append(d)
    
  with open(filename2) as f:
    dataset = json.load(f)
  for d in dataset:
    data.append(d)
  return data

if __name__ == "__main__":
  
  # opt_answer_datasets_len = get_alpaca("/workspace/vllm/examples/alpaca_opt13b_answer.json")
  # with open("/workspace/vllm/examples/alpaca_opt13b_answer_len.json", 'w') as json_file:
  #   json.dump(opt_answer_datasets_len, json_file, indent=2)
    
  data = union_dataset("/workspace/alpaca_opt13b_answer.json", "/workspace/vllm/examples/alpaca_opt13b_answer.json")
  with open("/workspace/vllm/examples/alpaca_opt13b_answer_union.json", 'w') as json_file:
    json.dump(data, json_file, indent=2)