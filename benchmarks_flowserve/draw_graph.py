import os
from typing import Dict, List
from matplotlib import pyplot as plt
def list_files_in_directory(directory):
    filepathes = []
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            filepathes.append(file_path)
            filenames.append(file)
    return filepathes, filenames

def process_res(filepathes, filenames):
    for filepath, filename in zip(filepathes, filenames):
        pd_type, input_len, output_len, reqs, num_requests = filename.split("_")
        if pd_type == "disagg":
            reqs = float(reqs)/2
        fixed_input_output = str(input_len) + "_" + str(output_len)
        if fixed_input_output not in fixed_input_output_table:
            pdtype_dict = {}
        else:
            pdtype_dict = fixed_input_output_table[fixed_input_output] 
        if pd_type not in pdtype_dict:
            reqs_dict = {}
        else:
            reqs_dict = pdtype_dict[pd_type]
        with open(filepath, "r") as fd:
            for line in fd.readlines():
                content = line.split("\n")[0].split(",")[-1].split(" ")
                data = content[3:]
                float_data = [float(item) for item in data]
                reqs_dict[float(reqs)] = float_data
                
        pdtype_dict[pd_type] = reqs_dict
        fixed_input_output_table[fixed_input_output] = pdtype_dict

def plot_func(c_average, d_average, c_p50, d_p50, c_p99, d_p99, name, fixed_input_output):
    plt.plot(label, c_average, label='colocate_average')
    plt.plot(label, d_average, label='disagg_average')
    plt.plot(label, c_p50, label='colocate_p50')
    plt.plot(label, d_p50, label='disagg_p50')
    plt.plot(label, c_p99, label='colocate_p99')
    plt.plot(label, d_p99, label='disagg_p99')
    plt.title(name + " under " + "input_output " + fixed_input_output)
    plt.xlabel('req/s')
    plt.ylabel('time(s)')
    plt.legend()
    plt.grid(True)
    plt.savefig("fig_" + name + "_under_" + "input_output_" + fixed_input_output + ".pdf", format='pdf', dpi=300)

    plt.show()

def plot_tbt_func(c_average_no_second, c_average_with, d_average_no_second, d_average_with, 
                  c_p50_no_second, c_p50_with, d_p50_no_second, d_p50_with,
                  c_p99_no_second, c_p99_with, d_p99_no_second, d_p99_with,
                  name, fixed_input_output):

    plt.plot(label, c_average_no_second, label='col_avg_no_second', color='b')
    plt.plot(label, c_average_with, label='col_avg_with', color='g')
    plt.plot(label, d_average_no_second, label='disagg_avg_no_second', color='r')
    plt.plot(label, d_average_with, label='disagg_avg_with', color='c')
    
    plt.plot(label, c_p50_no_second, label='col_p50_no_second', color='m')
    plt.plot(label, c_p50_with, label='col_p50_with', color='y')
    plt.plot(label, d_p50_no_second, label='disagg_p50_no_second', color='k')
    plt.plot(label, d_p50_with, label='disagg_p50_with', color='orange')
    
    plt.plot(label, c_p99_no_second, label='col_p99_no_second', color='purple')
    plt.plot(label, c_p99_with, label='col_p99_with', color='brown')
    plt.plot(label, d_p99_no_second, label='disagg_p99_no_second', color='pink')
    plt.plot(label, d_p99_with, label='disagg_p99_with', color='gray')
    
    plt.title(name + " under " + "input_output " + fixed_input_output)
    plt.xlabel('req/s')
    plt.ylabel('time(s)')
    plt.legend()
    plt.grid(True)
    plt.savefig("fig_" + name + "_under_" + "input_output_" + fixed_input_output +".pdf", format="pdf", dpi=300)

    plt.show()



if __name__ == "__main__":
    '''
    Assuming result folder structure:

    basename  
        dataset1
            baseline1
            baseline2
            ours
        dataset2
        dataset3
    '''
    fixed_input_output_table: Dict[str, Dict[str, Dict[str, List]]] = {}
    directory_path = '/Users/gaofz/Desktop/胡cunchen/Phd/LLM/dmem/cacheserve/e2e/vllm/colocate'
    filepathes, filenames = list_files_in_directory(directory_path)
    process_res(filepathes, filenames)
    directory_path = '/Users/gaofz/Desktop/胡cunchen/Phd/LLM/dmem/cacheserve/e2e/vllm/disagg'
    filepathes, filenames = list_files_in_directory(directory_path)
    process_res(filepathes, filenames)

    for fixed_input_output_key, fixed_input_output_value in fixed_input_output_table.items():
        label = []
        colocate_average_jct = []
        colocate_p50_jct = []
        colocate_p99_jct = []
        
        colocate_average_ttft = []
        colocate_p50_ttft = []
        colocate_p99_ttft = []

        colocate_average_tbt_n_sec = []
        colocate_p50_tbt_n_sec = []
        colocate_p99_tbt_n_sec = []
        
        colocate_average_tbt_w_sec = []
        colocate_p50_tbt_w_sec = []
        colocate_p99_tbt_w_sec = []

        disagg_average_jct = []
        disagg_p50_jct = []
        disagg_p99_jct = []
        
        disagg_average_ttft = []
        disagg_p50_ttft = []
        disagg_p99_ttft = []

        disagg_average_tbt_n_sec = []
        disagg_p50_tbt_n_sec = []
        disagg_p99_tbt_n_sec = []
        
        disagg_average_tbt_w_sec = []
        disagg_p50_tbt_w_sec = []
        disagg_p99_tbt_w_sec = []
        for pd_type, pd_type_value in fixed_input_output_value.items():
            sorted_by_key_value = {key: pd_type_value[key] for key  in sorted(pd_type_value.keys())}
            for req_label_key, req_lable_value in sorted_by_key_value.items():
                if pd_type == "colocate":
                    label.append(req_label_key)
                    colocate_average_jct.append(req_lable_value[0])
                    colocate_p50_jct.append(req_lable_value[1])
                    colocate_p99_jct.append(req_lable_value[2])
                    colocate_average_ttft.append(req_lable_value[3])
                    colocate_p50_ttft.append(req_lable_value[4])
                    colocate_p99_ttft.append(req_lable_value[5])
                    colocate_average_tbt_n_sec.append(req_lable_value[6])
                    colocate_p50_tbt_n_sec.append(req_lable_value[7])
                    colocate_p99_tbt_n_sec.append(req_lable_value[8])
                    colocate_average_tbt_w_sec.append(req_lable_value[9])
                    colocate_p50_tbt_w_sec.append(req_lable_value[10])
                    colocate_p99_tbt_w_sec.append(req_lable_value[11])
                else:
                    disagg_average_jct.append(req_lable_value[0])
                    disagg_p50_jct.append(req_lable_value[1])
                    disagg_p99_jct.append(req_lable_value[2])
                    disagg_average_ttft.append(req_lable_value[3])
                    disagg_p50_ttft.append(req_lable_value[4])
                    disagg_p99_ttft.append(req_lable_value[5])
                    disagg_average_tbt_n_sec.append(req_lable_value[6])
                    disagg_p50_tbt_n_sec.append(req_lable_value[7])
                    disagg_p99_tbt_n_sec.append(req_lable_value[8])
                    disagg_average_tbt_w_sec.append(req_lable_value[9])
                    disagg_p50_tbt_w_sec.append(req_lable_value[10])
                    disagg_p99_tbt_w_sec.append(req_lable_value[11])
        
        
        plot_func(colocate_average_jct, disagg_average_jct, colocate_p50_jct, disagg_p50_jct, colocate_p99_jct, disagg_p99_jct, "JCT", fixed_input_output_key)

        plot_func(colocate_average_ttft, disagg_average_ttft, colocate_p50_ttft, disagg_p50_ttft, colocate_p99_ttft, disagg_p99_ttft, "TTFT", fixed_input_output_key)


        plot_tbt_func(colocate_average_tbt_n_sec, colocate_average_tbt_w_sec, 
                disagg_average_tbt_n_sec, disagg_average_tbt_w_sec,
                colocate_p50_tbt_n_sec, colocate_p50_tbt_w_sec,
                disagg_p50_tbt_n_sec, disagg_p50_tbt_w_sec,
                colocate_p99_tbt_n_sec, colocate_p99_tbt_w_sec,
                disagg_p99_tbt_n_sec, disagg_p99_tbt_w_sec, "TBT With And WithOut Second Token", fixed_input_output_key)

            
        # fixed_output_reqs = str(pd_type) + "_" + str(output_len) + "_" + str(reqs)
