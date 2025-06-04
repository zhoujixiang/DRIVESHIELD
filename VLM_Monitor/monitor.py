import os
import re
import json

from graph import generate_scene_graph
from prompt import generate_vehicle_descriptions, get_collision_reasoning_with_image
from util import find_key_frames, get_closest_npc_3d, get_fp_collision, load_json_file, save_graph_pair_to_db

def run_collision_prediction_pipeline(base_path, key_frames, start_index=5, db_path="/bdata/usrdata/root/monitor_test/Bench2Drive/graph_pair_db.jsonl", dp_path_collision="/bdata/usrdata/root/monitor_test/Bench2Drive/graph_pair_db_collision.jsonl"):
    record_dir = os.path.join(base_path, 'record/')
    result_dir = os.path.join(base_path, 'result_graph/')
    prompt_dir = os.path.join(base_path, 'prompt_graph/')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)

    for id_frame in range(start_index, len(key_frames)):
        current_frame = int(os.path.splitext(key_frames[id_frame])[0])
        # 获取描述信息和图数据
        descriptions = generate_vehicle_descriptions(record_dir, current_frame)
        current_scene, past_scene, current_scene_json, past_scene_json = generate_scene_graph(record_dir, current_frame)
        metadata = {"path": current_scene_json}

        # 调用模型预测碰撞信息
        result = get_collision_reasoning_with_image(
            base_path, current_frame,current_scene, past_scene,
            descriptions["vehicle_description"],
            descriptions["vehicle_description_0.5s"],
            descriptions["vehicle_description_1.0s"],
            descriptions["vehicle_description_1.5s"],
            descriptions["vehicle_description_2.0s"]
        )
        
        # 保存结果文本
        id = f"{current_frame:04d}"
        result_path = os.path.join(result_dir, f"{id}.txt")
        with open(result_path, "w") as f:
            f.write(result)

        # 获取碰撞分数及高风险NPC
        score, dangerous_npc = get_fp_collision(result_path)
        if id_frame == len(key_frames) - 1:
            if score == 2 or score == 1:
                files = sorted([
                    f for f in os.listdir(record_dir)
                    if os.path.isfile(os.path.join(record_dir, f))
                    and re.fullmatch(r"\d{4}\.json", f)
                ])
                file = files[-1]
                file_path = os.path.join(record_dir, file)
                data = load_json_file(file_path)
                closest_npc_id, min_dist = get_closest_npc_3d(data)
                metadata["dangerous_npc"] = [closest_npc_id]
                save_graph_pair_to_db(current_scene, past_scene, metadata, dp_path_collision)

                    # 保存描述信息为 JSON 文件
            prompt_path = os.path.join(prompt_dir, f"{id}.json")
            with open(prompt_path, "w") as f:
                json.dump(descriptions, f, indent=2, ensure_ascii=False)
            break
        
        if score != 2 and score != 1:
            metadata["dangerous_npc"] = dangerous_npc
            save_graph_pair_to_db(current_scene, past_scene, metadata, db_path)

        # 保存描述信息为 JSON 文件
        prompt_path = os.path.join(prompt_dir, f"{id}.json")
        with open(prompt_path, "w") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    dir_path = "/bdata/usrdata/root/monitor_test/Bench2Drive/eval_bench2drive220_vad_traj/collision"
    dir_list = sorted(os.listdir(dir_path))
    for dir_index in range(len(dir_list)):
        name = dir_list[dir_index]
        base_path = os.path.join(dir_path, name)
        record_dir = os.path.join(base_path, 'record/')
        key_frames = find_key_frames(record_dir)
        print(key_frames)
        run_collision_prediction_pipeline(base_path, key_frames, start_index=0)
