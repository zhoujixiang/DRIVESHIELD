import os
import csv
import pandas as pd
def find_and_record_paths_in_order(root_folder, output_csv, step=1):
    all_file_paths = []
    for folder_name in sorted(os.listdir(root_folder)): 

        if "RouteScenario" in folder_name:
            scenario_path = os.path.join(root_folder, folder_name)

            if os.path.isdir(scenario_path):

                rgb_front_path = os.path.join(scenario_path, "bev")
                if os.path.isdir(rgb_front_path):

                    files = sorted(os.listdir(rgb_front_path))  
                    for i, filename in enumerate(files):

                        if i % step == 0:
                            file_path = os.path.join(rgb_front_path, filename)
                            if os.path.isfile(file_path): 
                                all_file_paths.append(file_path)


    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path"])  
        for path in all_file_paths:
            writer.writerow([path])

    print(f"记录完成！共记录了 {len(all_file_paths)} 个文件路径到 {output_csv}.")


import os
import csv

def write_normal_file_paths_to_csv(folder_path, output_csv, step=1, n=10):

    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

    file_paths.sort(key=lambda x: os.path.basename(x)) 
    selected_paths = file_paths[::step]


    remainder = len(selected_paths) % n
    if remainder != 0:
        selected_paths = selected_paths[:-remainder] 

 
    file_exists = os.path.exists(output_csv)
    
 
    with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        if not file_exists:
            writer.writerow(["path", "loss"])
        for path in selected_paths:
            writer.writerow([path, 0.0]) 

    print(f"所有文件路径及默认 loss 值已按照文件名顺序写入 {output_csv}，共 {len(selected_paths)} 条记录，确保是 {n} 的倍数。")




def write_abnormal_file_paths_to_csv(folder_path, output_csv, n, step =2):

    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    

    file_paths.sort(key=lambda x: os.path.basename(x)) 
    file_paths = file_paths[::step]

    total_rows = len(file_paths)
    crashed_values = [1 if i < n else 0 for i in range(total_rows)][::-1] 

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "loss", "crashed"])  
        for path, crashed in zip(file_paths, crashed_values):
            writer.writerow([path, 0.0, crashed])  

    print(f"所有文件路径、默认 loss 值及 crashed 标注已按照文件名顺序写入 {output_csv}")


def count_paths_all_true_positive(csv_file, n):

    df = pd.read_csv(csv_file)


    df = df[df["aggregation_method"] == "max"]


    df = df[df["seconds"] <= n]


    valid_paths = df.groupby("path")["true_positive_windows"].apply(lambda x: (x == 1).all())

    count = valid_paths.sum()

    return count



if __name__ == '__main__':


  
  
  

    base_path = "/bdata/usrdata/root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/collision_need_judge_2"


    folders = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))  
    ]


    folders_sorted = sorted(folders, key=lambda x: os.path.basename(x))

    for folder in folders_sorted:
        folder_path = os.path.join(folder, 'rgb_front')   
        output_csv = os.path.join(folder, 'driving_log_bev.csv')  
  
  
  
  
        write_abnormal_file_paths_to_csv(folder_path, output_csv,10, 1)
  

  
  
  
  
  
  