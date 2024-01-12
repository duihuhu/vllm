import requests
import time
def post_init_decode_prefill() -> requests.Response:
    api_url = "http://127.0.0.1:7001/covert_mprefill_mdecode_exec" 
    headers = {"User-Agent": "Test Client"}
    start_time = time.time()
    response = requests.post(api_url, headers=headers, stream=True)
    end_time = time.time()
    print("post_init_decode_prefill ", start_time, end_time)
    return response
