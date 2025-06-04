
import re
import json
import math
from PIL import Image
import os,io
import base64

def extract_reasons(folder_path):
    reasons = []
    unreadable_files = []

    with open(folder_path, 'r') as file:
        for line in file:
            if '"reason"' in line:
                file_has_reason = True
  
                match = re.search(r'"reason"\s*:\s*"([^"]*)"', line)
                if match:
                    reasons.append(match.group(1))
                break   
    return reasons

def save_graph_pair_to_db(current_graph, past_graph, metadata, filepath="graph_pair_db.jsonl"):
    entry = {
        "current_graph": current_graph,
        "past_graph": past_graph,
        "metadata": metadata   
    }
    with open(filepath, "a") as f:
        f.write(json.dumps(entry) + "\n")


def compute_longitudinal_safe_distance(speed, reaction_time=0.001, friction=0.5, gravity=9.81, vehicle_length = 4.8):
    """
    计算纵向安全包络（前后安全距离）
    
    参数:
        speed (float): 车辆当前速度 (m/s)
        reaction_time (float): 反应时间 (s)
        friction (float): 路面摩擦系数
        gravity (float): 重力加速度 (m/s^2)
        
    返回:
        safe_distance (float): 纵向安全包络距离 (m)
    """
    braking_distance = (speed ** 2) / (2 * friction * gravity)
    reaction_distance = speed * reaction_time
    safe_distance = braking_distance + reaction_distance + vehicle_length
    return safe_distance

  
def compute_lateral_safe_distance(speed, vehicle_width=2.16, turn_radius=3.68):
    """
    计算横向安全包络（侧向安全距离）
    
    参数:
        vehicle_width (float): 车辆宽度 (m)
        speed (float): 车辆当前速度 (m/s)
        turn_radius (float): 车辆的最小转弯半径 (m)
        
    返回:
        lateral_safe_distance (float): 横向安全包络距离 (m)
    """
    lateral_offset = (speed ** 2) / (2 * turn_radius)   
    return vehicle_width + 2 * lateral_offset

def load_json_file(file_path):
   
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def find_key_frames(record_dir, start=30, end = 5, step=5):
   
    key_frame = []

  
    json_files = sorted([f for f in os.listdir(record_dir) if f.endswith(".json") and not (f.endswith("_visible.json") or f.endswith("_true.json"))])

  
    for i in range(len(json_files) - end - 1, start - 1, -step):
        json_file = json_files[i]
        file_path = os.path.join(record_dir, json_file)
        data = load_json_file(file_path)

  
        ego_speed = data.get("ego_vehicle", {}).get("speed", 0.0)

  
        longitudinal_safe_distance = compute_longitudinal_safe_distance(ego_speed)
        lateral_safe_distance = compute_lateral_safe_distance(ego_speed)

  
        for key in data:
            if key.startswith("npc"):
                npc = data[key]
                lat_dist = abs(npc.get("lateral_distance", 0.0))
                lon_dist = abs(npc.get("longitudinal_distance", 0.0))

                if (lat_dist < lateral_safe_distance or lon_dist < longitudinal_safe_distance) and npc.get("longitudinal_distance") > -1:
                    key_frame.append(json_file)
                    break   
    key_frame.sort()
    return key_frame

def get_fp_collision(file_path):
    score = None
    dangerous_npcs = []

    with open(file_path, 'r') as file:
        for line in file:
  
            if "risk_score" in line:
                match = re.search(r'"risk_score"\s*:\s*(\[\s*(\d+)\s*\]|(\d+))', line)
                if match:
                    score_str = match.group(2) if match.group(2) is not None else match.group(3)
                    score = int(score_str)

  
            if "dangerous_npc" in line:
  
                array_match = re.search(r'"dangerous_npc"\s*:\s*\[(.*?)\]', line)
                if array_match:
                    dangerous_npcs = re.findall(r'"([^"]+)"', array_match.group(1))
                else:
  
                    single_match = re.search(r'"dangerous_npc"\s*:\s*"([^"]+)"', line)
                    if single_match:
                        dangerous_npcs = [single_match.group(1)]

    return score, dangerous_npcs

def get_closest_npc_3d(data):
    ego_loc = data.get("ego_vehicle", {}).get("location", {})
    if not ego_loc:
        return None

    min_dist = float("inf")
    closest_npc_id = None

    ego_x, ego_y, ego_z = ego_loc["x"], ego_loc["y"], ego_loc["z"]

    for key, value in data.items():
        if key.startswith("npc") and "location" in value:
            npc_loc = value["location"]
            npc_x, npc_y, npc_z = npc_loc["x"], npc_loc["y"], npc_loc["z"]
            dist = math.sqrt((npc_x - ego_x) ** 2 + (npc_y - ego_y) ** 2 + (npc_z - ego_z) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_npc_id = key.upper()

    return closest_npc_id, min_dist

def compress_and_encode(image_path, target_width=200):
    img = Image.open(image_path)
    w, h = img.size
    ratio = target_width / w
    target_height = int(h * ratio)

  
    resample = getattr(Image, 'Resampling', Image).LANCZOS

    resized_img = img.resize((target_width, target_height), resample)

    buffer = io.BytesIO()
    resized_img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_float(value, precision=2):
    return round(float(value), precision) 