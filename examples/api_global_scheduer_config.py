host_ip = "127.0.0.1"
global_scheduler_port = 8000
add_reqs_url = f"http://{host_ip}:{global_scheduler_port}/add_reqs"

forward_mprefill_url = "http://%s:%s/mprefill_add"

client_port = 9000
forward_res_url = "http://{host_ip}::{client_port}/response"


forward_mpd_url = "http://%s:%s/mpd_add"


session_num = 1
