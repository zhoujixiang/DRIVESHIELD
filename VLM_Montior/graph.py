  
  
  
from itertools import combinations
from collections import Counter
from collections import defaultdict
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import json
import os
import re
import cv2

from util import load_json_file

def same_sign(a, b):
    return a * b > 0 or (a == 0 and b == 0)

def is_graph_entry_match(entry_a, entry_b):
    return (
        entry_a["lane_difference"] == entry_b["lane_difference"]
        and entry_a["relative_position"] == entry_b["relative_position"]
        and same_sign(entry_a["longitudinal_distance"], entry_b["longitudinal_distance"])
        and same_sign(entry_a["lateral_distance"], entry_b["lateral_distance"])
  
  
    )


def compare_pair_graph(target_pair, pair):
    """
    fp_tn_candidate = {
    0: [{"index": 2, "path": "scene_01.json"}, {"index": 5, "path": "scene_02.json"}],
    1: [{"index": 1, "path": "scene_02.json"}, {"index": 7, "path": "scene_03.json"}],
    ...
}
    """
    fp_tn_candidate = {}
    target_current_graph = target_pair["current_graph"]
    target_past_graph = target_pair["past_graph"]
    pair_current_graph_orginal = pair["current_graph"]
    pair_past_graph_orginal = pair["past_graph"]
    pair_current_graph = []
    pair_past_graph = []
  
    dangerous_npc_map = {}   
    raw_dangerous_npc = pair["metadata"].get("dangerous_npc", [])
    dangerous_npc_list = []
    if isinstance(raw_dangerous_npc, str):
        dangerous_npc_list = [npc.strip() for npc in raw_dangerous_npc.split(",")]
    elif isinstance(raw_dangerous_npc, list):
        for item in raw_dangerous_npc:
            if isinstance(item, str) and "," in item:
  
                dangerous_npc_list.extend([npc.strip() for npc in item.split(",")])
            else:
                dangerous_npc_list.append(item)

    for idx, dangerous_npc in enumerate(dangerous_npc_list):
        suffix = dangerous_npc[3:]
        if not (isinstance(suffix, str) and suffix.isdigit()):
  
            return {}
        npc_id = int(suffix) - 1
        pair_current_graph.append(pair_current_graph_orginal[npc_id])
        pair_past_graph.append(pair_past_graph_orginal[npc_id])
        dangerous_npc_map[idx] = dangerous_npc 
    for agent_i in range(len(target_current_graph)):
        fp_tn_candidate[agent_i] = []
        target_past_graph_gent_i = target_past_graph[agent_i]
        for candidate in pair_past_graph:
            if is_graph_entry_match(candidate, target_past_graph_gent_i):#过去时刻相对场景图相同
                candidate_index = pair_past_graph.index(candidate)
                candidate_current = pair_current_graph[candidate_index]
                target_current_graph_gent_i = target_current_graph[agent_i]
                if is_graph_entry_match(candidate_current, target_current_graph_gent_i):#当前时刻相对场景图相同
                    target_relative_lateral_distance = abs(target_current_graph_gent_i["lateral_distance"]) - abs(target_past_graph_gent_i["lateral_distance"])
                    target_relative_longitudinal_distance = abs(target_current_graph_gent_i["longitudinal_distance"]) - abs(target_past_graph_gent_i["longitudinal_distance"])
                    candidate_relative_lateral_distance = abs(candidate_current["lateral_distance"]) - abs(candidate["lateral_distance"])
                    candidate_relative_longitudinal_distance = abs(candidate_current["longitudinal_distance"]) - abs(candidate["longitudinal_distance"])
                    if same_sign(target_relative_lateral_distance, candidate_relative_lateral_distance) and same_sign(target_relative_longitudinal_distance, candidate_relative_longitudinal_distance):#相对位移相同,是靠近还是远离ego
  
                        if abs(target_current_graph_gent_i["lateral_distance"] - candidate_current["lateral_distance"]) < 1 and abs(target_current_graph_gent_i["longitudinal_distance"] - candidate_current["longitudinal_distance"]) < 2 and\
                            abs(target_current_graph_gent_i["lateral_speed"] - candidate_current["lateral_speed"]) < 3 and abs(target_current_graph_gent_i["longitudinal_speed"] - candidate_current["longitudinal_speed"]) < 3:
                            fp_tn_candidate[agent_i].append({
                            "index": candidate_index,
                            "path": pair["metadata"]["path"],
                            "dangerous_npc": dangerous_npc_map[candidate_index],   
                        })
    
    return fp_tn_candidate   

def compare_pair_graph_tp_graph(target_pair, pair):
    """
    fp_tn_candidate = {
    0: [{"index": 2, "path": "scene_01.json"}, {"index": 5, "path": "scene_02.json"}],
    1: [{"index": 1, "path": "scene_02.json"}, {"index": 7, "path": "scene_03.json"}],
    ...
}
    """
    fp_tn_candidate = {}
    target_current_graph = target_pair["current_graph"]
    target_past_graph = target_pair["past_graph"]
    pair_current_graph_orginal = pair["current_graph"]
    pair_past_graph_orginal = pair["past_graph"]
    pair_current_graph = []
    pair_past_graph = []
  
    dangerous_npc_map = {}   
    raw_dangerous_npc = pair["metadata"].get("dangerous_npc", [])
    dangerous_npc_list = []
    if isinstance(raw_dangerous_npc, str):
        dangerous_npc_list = [npc.strip() for npc in raw_dangerous_npc.split(",")]
    elif isinstance(raw_dangerous_npc, list):
        for item in raw_dangerous_npc:
            if isinstance(item, str) and "," in item:
  
                dangerous_npc_list.extend([npc.strip() for npc in item.split(",")])
            else:
                dangerous_npc_list.append(item)

    for idx, dangerous_npc in enumerate(dangerous_npc_list):
        suffix = dangerous_npc[3:]
        if not (isinstance(suffix, str) and suffix.isdigit()):
  
            return {}
        npc_id = int(suffix) - 1
        pair_current_graph.append(pair_current_graph_orginal[npc_id])
        pair_past_graph.append(pair_past_graph_orginal[npc_id])
        dangerous_npc_map[idx] = dangerous_npc 
  
  
  
  
  
  
  
  
  
  
  
    
    for agent_i in range(len(target_current_graph)):
        fp_tn_candidate[agent_i] = []
        target_past_graph_gent_i = target_past_graph[agent_i]
        for candidate in pair_past_graph:
            if is_graph_entry_match(candidate, target_past_graph_gent_i):#过去时刻相对场景图相同
                candidate_index = pair_past_graph.index(candidate)
                candidate_current = pair_current_graph[candidate_index]
                target_current_graph_gent_i = target_current_graph[agent_i]
                if is_graph_entry_match(candidate_current, target_current_graph_gent_i):#当前时刻相对场景图相同
                    target_relative_lateral_distance = abs(target_current_graph_gent_i["lateral_distance"]) - abs(target_past_graph_gent_i["lateral_distance"])
                    target_relative_longitudinal_distance = abs(target_current_graph_gent_i["longitudinal_distance"]) - abs(target_past_graph_gent_i["longitudinal_distance"])
                    candidate_relative_lateral_distance = abs(candidate_current["lateral_distance"]) - abs(candidate["lateral_distance"])
                    candidate_relative_longitudinal_distance = abs(candidate_current["longitudinal_distance"]) - abs(candidate["longitudinal_distance"])
                    if same_sign(target_relative_lateral_distance, candidate_relative_lateral_distance) and same_sign(target_relative_longitudinal_distance, candidate_relative_longitudinal_distance):#相对位移相同,是靠近还是远离ego
  
                        if abs(target_current_graph_gent_i["lateral_distance"] - candidate_current["lateral_distance"]) < 1 and abs(target_current_graph_gent_i["longitudinal_distance"] - candidate_current["longitudinal_distance"]) < 3:
                            fp_tn_candidate[agent_i].append({
                            "index": candidate_index,
                            "path": pair["metadata"]["path"],
                            "dangerous_npc": dangerous_npc_map[candidate_index],   
                        })
    
    return fp_tn_candidate   

def find_most_similar_graph_pair(cur_now, past_now, db_path="/bdata/usrdata/zjx/monitor_test/Bench2Drive/graph_pair_db.jsonl"):
    if not os.path.exists(db_path):
        return None
    target_pair = {
        "current_graph": cur_now,
        "past_graph": past_now
    }
    merged_result = defaultdict(list)
    with open(db_path, "r") as f:
        for line in f:
            pair = json.loads(line)
            match = compare_pair_graph(target_pair, pair)
            if match and any(match.values()): 
                for key, value in match.items():
                    merged_result[key].extend(value)

    return dict(merged_result)

def find_most_similar_graph_pair_tp_graph(cur_now, past_now, tp_graph, db_path="/bdata/usrdata/zjx/monitor_test/Bench2Drive/graph_pair_db.jsonl"):
    if not os.path.exists(db_path):
        return None
    target_pair = {
        "current_graph": cur_now,
        "past_graph": past_now
    }
    merged_result = defaultdict(list)
    with open(db_path, "r") as f:
        for line in f:
            pair = json.loads(line)
            match = compare_pair_graph(target_pair, pair)
            if match and any(match.values()): 
                for key, value in match.items():
                    merged_result[key].extend(value)

    return dict(merged_result)

def compute_similarity(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert("L").resize((256, 256)))
    img2 = np.array(Image.open(img2_path).convert("L").resize((256, 256)))
    return ssim(img1, img2)

def compute_multi_view_similarity(current_left_image, current_front_image, current_right_image,
                                  front_left_image, front_image, front_right_image):
    total_score = 0.0

    image_pairs = [
        (current_left_image, front_left_image),
        (current_front_image, front_image),
        (current_right_image, front_right_image)
    ]

    for cur_img_path, cmp_img_path in image_pairs:
        if os.path.exists(cur_img_path) and os.path.exists(cmp_img_path):
            try:
                score = compute_similarity(cur_img_path, cmp_img_path)
                total_score += score
            except Exception as e:
                print(f"Error computing similarity for view: {e}")
                continue

    return total_score


def select_best_matches(merged_result, current_left_image, current_front_image, current_right_image):
    path_counter = Counter()
    path_to_agents = defaultdict(list)

    for agent_id, candidates in merged_result.items():
        for entry in candidates:
            path = entry["path"]
            path_counter[path] += 1
            path_to_agents[path].append(agent_id)

  
    common_paths = {path: count for path, count in path_counter.items() if count > 1}

    selected = []
    common_best_record = {}
    unique_best_record = {}
    record = {}
    if common_paths:
        max_count = max(common_paths.values())   
        best_paths = [path for path, count in common_paths.items() if count == max_count]   
        print(f"Best paths: {best_paths}")
        path_similarity_scores = {}
        for path in best_paths:   
            base_dir = os.path.dirname(os.path.dirname(path))
            filename = os.path.basename(path)   
            number_str = os.path.splitext(filename)[0]
            front_left_dir = os.path.join(base_dir, 'rgb_front_left/')
            front_left_image = os.path.join(front_left_dir, f"{number_str}.png")
            front_dir = os.path.join(base_dir, 'rgb_front/')
            front_image = os.path.join(front_dir, f"{number_str}.png")
            front_right_dir = os.path.join(base_dir, 'rgb_front_right/')
            front_right_image = os.path.join(front_right_dir, f"{number_str}.png")
            path_similarity_scores[path] = compute_multi_view_similarity(current_left_image, current_front_image, current_right_image, front_left_image, front_image, front_right_image)
        best_path = max(path_similarity_scores.items(), key=lambda x: x[1])[0]
        for agent_id in path_to_agents[best_path]:
            selected.append(agent_id)
            for entry in merged_result[agent_id]:
                if entry["path"] == best_path:
                    record = {
                        "path": entry["path"],
                        "dangerous_npc": entry["dangerous_npc"],
                        "similarity_score": path_similarity_scores[best_path]
                    }
                    common_best_record[agent_id] = record
                    break

  
    for agent_id, candidates in merged_result.items():
        best_entry = None
        if agent_id in selected or not candidates:
            continue
        best_score = -1
        for entry in candidates:
            base_dir = os.path.dirname(os.path.dirname(entry["path"]))
            filename = os.path.basename(entry["path"])   
            number_str = os.path.splitext(filename)[0]
            front_left_dir = os.path.join(base_dir, 'rgb_front_left/')
            front_left_image = os.path.join(front_left_dir, f"{number_str}.png")
            front_dir = os.path.join(base_dir, 'rgb_front/')
            front_image = os.path.join(front_dir, f"{number_str}.png")
            front_right_dir = os.path.join(base_dir, 'rgb_front_right/')
            front_right_image = os.path.join(front_right_dir, f"{number_str}.png")
            try:
                sim_score = compute_multi_view_similarity(current_left_image, current_front_image, current_right_image, front_left_image, front_image, front_right_image)
                if sim_score > best_score:
                    best_score = sim_score
                    best_entry = entry 
            except Exception as e:
                print(f"Error comparing images: {e}")
        if best_entry:
            best_entry["similarity_score"] = best_score
            unique_best_record[agent_id] = best_entry
    print("zjx")
    print(common_best_record, unique_best_record)
    return common_best_record, unique_best_record

def extract_relationship_graph(data):
    relationships = []
    for entity, attributes in data.items():
  
        if entity != "ego_vehicle":
            relationships.append({
  
  
                "lane_difference": attributes["change_lane_number"],
                "relative_position": "left" if attributes["left_right_lane"] == "right" else "right",
  
                "lateral_distance": attributes["lateral_distance"],
                "longitudinal_distance": attributes["longitudinal_distance"],
                "lateral_speed": data["ego_vehicle"]["self_lateral_speed"] - attributes["self_lateral_speed"],
                "longitudinal_speed": data["ego_vehicle"]["self_longitudinal_speed"] - attributes["self_longitudinal_speed"],
            })
    return relationships

def generate_scene_graph(record_dir, current_frame, step=20):
    past_frame = current_frame - step
    current_frame = f"{current_frame:04d}"
    past_frame = f"{past_frame:04d}"
    current_json = os.path.join(record_dir, f"{current_frame}.json")
    past_json = os.path.join(record_dir, f"{past_frame}.json")
    current_data = load_json_file(current_json)
    past_data = load_json_file(past_json)
    current_graph = extract_relationship_graph(current_data)
    past_graph = extract_relationship_graph(past_data)
    return current_graph, past_graph, current_json, past_json