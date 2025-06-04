import os
import csv
import pandas as pd
def find_and_record_paths_in_order(root_folder, output_csv, step=1):
    """
    遍历root_folder 下的 'RouteScenario' 文件夹，按指定步长记录 rgb_front 文件夹中的文件路径到 CSV。

    参数：
    -root_folder: 根文件夹路径
    - output_csv: 输出的 CSV 文件路径
    - step: 步长，每隔 step 个文件记录一个路径
    """
    # 用于存储结果的列表
    all_file_paths = []

    # 遍历root_folder 下的直接子文件夹
    for folder_name in sorted(os.listdir(root_folder)):  # 按文件夹名排序
        # 检查文件夹名是否包含 'RouteScenario'
        if "RouteScenario" in folder_name:
            scenario_path = os.path.join(root_folder, folder_name)
            # 确保路径是一个目录
            if os.path.isdir(scenario_path):
                # 检查是否有 rgb_front 子文件夹
                rgb_front_path = os.path.join(scenario_path, "bev")
                if os.path.isdir(rgb_front_path):
                    # 获取 rgb_front 文件夹中的所有文件，并排序
                    files = sorted(os.listdir(rgb_front_path))  # 按文件名排序
                    for i, filename in enumerate(files):
                        # 按步长筛选文件
                        if i % step == 0:
                            file_path = os.path.join(rgb_front_path, filename)
                            if os.path.isfile(file_path):  # 确保是文件而非文件夹
                                all_file_paths.append(file_path)

    # 将文件路径写入 CSV 文件
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path"])  # 写入表头
        for path in all_file_paths:
            writer.writerow([path])

    print(f"记录完成！共记录了 {len(all_file_paths)} 个文件路径到 {output_csv}.")

#用于生成normal的驾驶记录文件drving_log.csv
import os
import csv

def write_normal_file_paths_to_csv(folder_path, output_csv, step=1, n=10):
    """
    将指定文件夹下的所有文件路径按照文件名排序后写入 CSV 文件，并确保最终行数是 n 的倍数，删除多余部分。

    Args:
        folder_path (str): 要扫描的文件夹路径。
        output_csv (str): 输出的 CSV 文件路径。
        step (int): 选择文件的步长，默认 1（即所有文件）。
        n (int): 结果行数必须是 n 的倍数，默认 10。
    """
    # 获取所有文件路径
    file_paths = []
    forroot, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)

    # 按文件名排序
    file_paths.sort(key=lambda x: os.path.basename(x))  # 基于文件名排序
    selected_paths = file_paths[::step]

    # **确保 selected_paths 数量是 n 的倍数**
    remainder = len(selected_paths) % n
    if remainder != 0:
        selected_paths = selected_paths[:-remainder]  # 移除多余的行

    # 检查输出文件是否已存在
    file_exists = os.path.exists(output_csv)
    
    # 写入 CSV 文件
    with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # 仅在文件不存在时写入表头
        if not file_exists:
            writer.writerow(["path", "loss"])
        for path in selected_paths:
            writer.writerow([path, 0.0])  # 默认 loss 值为 0.0

    print(f"所有文件路径及默认 loss 值已按照文件名顺序写入 {output_csv}，共 {len(selected_paths)} 条记录，确保是 {n} 的倍数。")



#用于生成abnormal的驾驶记录文件driving_log.csv
def write_abnormal_file_paths_to_csv(folder_path, output_csv, n, step =2):
    """
    将指定文件夹下的所有文件路径按照文件名排序后写入 CSV 文件，并添加 'loss' 和 'crashed' 列。
    'loss' 列默认值为 0.0，'crashed' 列从最后一行往前标注 n 行为 1，其余为 0。

    Args:
        folder_path (str): 要扫描的文件夹路径。
        output_csv (str): 输出的 CSV 文件路径。
        n (int): 从最后一行往前标注 'crashed' 列为 1 的行数。
    """
    # 获取所有文件路径
    file_paths = []
    forroot, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    
    # 按文件名排序
    file_paths.sort(key=lambda x: os.path.basename(x))  # 基于文件名排序
    file_paths = file_paths[::step]
    # 初始化 crashed 列的值
    total_rows = len(file_paths)
    crashed_values = [1 if i < n else 0 for i in range(total_rows)][::-1]  # 从最后一行开始标注 1

    # 将文件路径、loss 和 crashed 写入 CSV 文件
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["path", "loss", "crashed"])  # 添加表头，包括 'path', 'loss', 'crashed'
        for path, crashed in zip(file_paths, crashed_values):
            writer.writerow([path, 0.0, crashed])  # 默认 loss 值为 0.0

    print(f"所有文件路径、默认 loss 值及 crashed 标注已按照文件名顺序写入 {output_csv}")


def count_paths_all_true_positive(csv_file, n):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 只保留 aggregation_method == "mean" 的行
    df = df[df["aggregation_method"] == "max"]

    # 只保留 seconds <= n 的数据
    df = df[df["seconds"] <= n]

    # 对每个 path 检查是否所有 seconds ≤ n 的 false_negative_windows 都是 1
    valid_paths = df.groupby("path")["true_positive_windows"].apply(lambda x: (x == 1).all())

    # 统计满足条件的 path 数量
    count = valid_paths.sum()

    return count



if __name__ == '__main__':

    # # # 用于生成训练数据
    #root_folder = "/bdata/usrdata/  root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/normal/train"  # 根文件夹路径
    # output_csv = "/bdata/usrdata/  root/monitor_test/selforacle_rebuild/train_output_fps10_bev.csv"  # 输出 CSV 文件名
    # find_and_record_paths_in_order(root_folder, output_csv,1)
    # 用于生成fp和tn的测试数据
    base_path = "/bdata/usrdata/  root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/collision_need_judge_2"

    # 获取所有子文件夹路径
    folders = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))  # 检查是否是文件夹
    ]

    # 按文件夹名称排序
    folders_sorted = sorted(folders, key=lambda x: os.path.basename(x))
    # 遍历每个文件夹，处理其中的 rgb_front 文件夹
    for folder in folders_sorted:
        folder_path = os.path.join(folder, 'rgb_front')  # rgb_front 文件夹路径
        output_csv = os.path.join(folder, 'driving_log_bev.csv')  # 输出 CSV 路径
        # output_csv = "/bdata/usrdata/  root/monitor_test/selforacle_rebuild/driving_log_bev.csv"
        # if os.path.exists(output_csv):
        #     os.remove(output_csv)
        # write_normal_file_paths_to_csv(folder_path, output_csv, step=1, n=5)
        write_abnormal_file_paths_to_csv(folder_path, output_csv,10, 1)
    # # import pandas as pd

    # # # 读取 CSV 文件
    # # 示例调用
    # csv_file = "/home/  root/ase22/selforacle_rebuild/test_data_results_tp_fn.csv"  # 替换为你的 CSV 文件路径
    # n = 1  # 目标 seconds
    # result = count_paths_all_true_positive(csv_file, n)
    # print(f"在 seconds={n} 时，其对应的 path 之前 (seconds < {n}) 有 false_negative_windows=1 的 path 数量: {result}")