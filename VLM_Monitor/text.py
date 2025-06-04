import os

def rename_result_folders(root_dir):
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            result_path = os.path.join(subdir_path, "result_graph")
            new_result_path = os.path.join(subdir_path, "result_graph_risk_2")
            if os.path.exists(result_path) and os.path.isdir(result_path):
                os.rename(result_path, new_result_path)
                print(f"Renamed: {result_path} -> {new_result_path}")
            else:
                print(f"No 'result' folder in {subdir_path}")

# 使用方法：替换为你的实际目录
root_folder = "/bdata/usrdata/root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/collision_need_to_judge"
rename_result_folders(root_folder)