import json
def get_alpaca(dataset_path):
    with open(dataset_path) as f:
        dataset = json.load(f)
        
    alpaca_data = []
    for data in dataset:
        if data['stop_reason'] == "stop":
            alpaca_data.append((data["instruction"], data["input"], data["output"]))
    return alpaca_data
  
if __name__ == "__main__":
  opt_answer_datasets_len = get_alpaca("/workspace/alpaca_opt13b_answer.json")
  with open("/workspace/alpaca_opt13b_answer_len.json", 'w') as json_file:
    json.dump(opt_answer_datasets_len, json_file, indent=2)