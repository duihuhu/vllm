import os
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np 
def plot_breakdown(values, xlabel, labels, fixed_output_reqs_key, colors, name):
    # 创建直方图
    fig, ax = plt.subplots()
    # 创建堆积柱状图
    for idx, lst in enumerate(values):
        if name == "ratio":
            total = sum(lst)
            proportions = [x / total for x in lst]
        else:
            proportions = lst
        bottom = 0
        for i, prop in enumerate(proportions):
            if idx == 0:
                ax.bar(idx, prop, bottom=bottom, color=colors[i], edgecolor='grey', label=labels[i])
            else:
                ax.bar(idx, prop, bottom=bottom, color=colors[i], edgecolor='grey')
            bottom += prop
    # 添加标签和标题
    if name == "ratio":
        ax.set_ylabel('Proportion')
    else:
        ax.set_ylabel('time(s)')
    ax.set_title("fixed_output_reqs " + fixed_output_reqs_key)
    ax.legend(loc='upper left')
    # 设置x轴刻度
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([xlabel[i] for i in range(len(values))])
    # 显示图表
    plt.savefig("fig" + "_under_fixed_" + "output_reqs " + fixed_output_reqs_key + "_" + name  + ".pdf", format="pdf", dpi=300)

    plt.show()
    
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
        pd_type, input_len, output_len, reqs, num_requests, *other = filename.split("_")
        filelabel = '_'.join(other)
        if pd_type == "disagg":
            reqs = float(reqs)/2
        fixed_output_reqs = str(output_len) + "_" + str(reqs)
        if fixed_output_reqs not in fixed_output_reqs_table:
            input_len_dict = {}
        else:
            input_len_dict = fixed_output_reqs_table[fixed_output_reqs]
            
        if input_len not in input_len_dict:
            filename_dict = {}
        else:
            filename_dict = input_len_dict[input_len]
        if filelabel not in filename_dict:
            request_id_dict = {}
        else:
            request_id_dict = filename_dict[filelabel]
            
        with open(filepath, "r") as fd:
            for line in fd.readlines():
                content = line.split("\n")[0].split(" ")
                request_id = content[-2]
                data = content[-1]
                request_id_dict[request_id] = data
        filename_dict[filelabel] = request_id_dict
        input_len_dict[input_len] = filename_dict
        fixed_output_reqs_table[fixed_output_reqs] = input_len_dict

# 指定你想要遍历的目录
fixed_output_reqs_table: Dict[str, Dict[str, Dict[str, List]]] = {}
directory_path = '/Users/gaofz/Desktop/胡cunchen/Phd/LLM/dmem/cacheserve/e2e/vllm/breakdown/disagg'
filepathes, filenames = list_files_in_directory(directory_path)
process_res(filepathes, filenames)
res_fixed_output_reqs_table = {}
for fixed_output_reqs_key, fixed_output_reqs_value in fixed_output_reqs_table.items():
    input_dict = {}
    for key, value in fixed_output_reqs_value.items():
        lats = []
        lats_allocate_kv = []
        lats_transfer_kv = [] 
        lats_exec_model = [] 
        p_send_query = value["prefill_send_query_kv_to_decode.txt"]
        p_add_kv_request = value["prefill_add_kv_request.txt"]
        d_add_runing = value["decode_add_request_to_running.txt"]
        d_finished = value["decode_finished_reqs.txt"]
        for req_id, value in p_send_query.items():
            p_add = p_add_kv_request[req_id]
            d_add_run = d_add_runing[req_id]
            d_fin = d_finished[req_id]
            lat_allocate_kv = float(p_add) - float(value)
            lat_transfer_kv = float(d_add_run) - float(p_add)
            lat_exec_model = float(d_fin) - float(d_add_run)
            lats_allocate_kv.append(lat_allocate_kv)
            lats_transfer_kv.append(lat_transfer_kv)
            lats_exec_model.append(lat_exec_model)
        input_dict[int(key)] = [np.average(lats_allocate_kv), np.average(lats_transfer_kv), np.average(lats_exec_model)]
    res_fixed_output_reqs_table[fixed_output_reqs_key] = input_dict

for fixed_output_reqs_key, fixed_output_reqs_value in res_fixed_output_reqs_table.items():
    sorted_by_key_value = {key: fixed_output_reqs_value[key] for key  in sorted(fixed_output_reqs_value.keys())}
    xlabel = []
    labels = ['avg_allocate_kv', 'avg_transfer_kv', 'avg_exec_model']
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    values = [] 
    for key, value in sorted_by_key_value.items():
        xlabel.append(int(key))
        # 计算每个列表的总和和占比
        values.append(value)
    
    plot_breakdown(values=values, xlabel=xlabel, labels=labels, fixed_output_reqs_key=fixed_output_reqs_key, colors=colors, name="ratio")
    
    plot_breakdown(values=values, xlabel=xlabel, labels=labels, fixed_output_reqs_key=fixed_output_reqs_key, colors=colors, name="value")
    # fig, ax1 = plt.subplots()
    # # 创建堆积柱状图
    # for idx, lst in enumerate(values):
    #     bottom = 0
    #     for i, value in enumerate(lst):
    #         if idx == 0:
    #             ax1.bar(idx, value, bottom=bottom, color=colors[i], edgecolor='grey', label=labels[i])
    #         else:
    #             ax1.bar(idx, value, bottom=bottom, color=colors[i], edgecolor='grey')
    #         bottom += value
    # # 添加标签和标题
    # ax1.set_ylabel('Proportion')
    # ax1.set_title("fixed_output_reqs " + fixed_output_reqs_key)
    # ax1.legend(loc='upper left')
    # # 设置x轴刻度
    # ax1.set_xticks(range(len(values)))
    # ax1.set_xticklabels([xlabel[i] for i in range(len(values))])
    # # 显示图表
    # plt.show()
