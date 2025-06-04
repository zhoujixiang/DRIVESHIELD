import json
import os
import base64
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
import io
from graph import find_most_similar_graph_pair, select_best_matches,find_most_similar_graph_pair_tp_graph
from util import extract_reasons, format_float, load_json_file


def generate_npc_prompt_old(npc_key, npc_data, ego_data):
    # Determine relative position (left or right)
    relative_side = "left" if npc_data["left_right_lane"] == "right" else "right"
    # # Determine if the NPC is in front or behind the ego vehicle
    # front_back = "front" if npc_data["longitudinal_distance"] > 0 else "behind"
    if npc_data["longitudinal_distance"] > 4.8:  
        front_back = "in front of ego"
        longitudinal_distance = abs(npc_data["longitudinal_distance"]) - 4.8
        distance_description = f"with NPC's rear {format_float(longitudinal_distance)} m in front of Ego's front"
    elif npc_data["longitudinal_distance"] < -4.8:  
        front_back = "behind ego"
        longitudinal_distance = abs(npc_data["longitudinal_distance"]) - 4.8
        distance_description = f"with NPC's front {format_float(longitudinal_distance)} m beind Ego's rear"
    elif npc_data["longitudinal_distance"] > 0.2 and npc_data["longitudinal_distance"] < 4.8:  
        front_back = "partially beside and slightly in front of ego"
        longitudinal_distance = abs(npc_data["longitudinal_distance"])  
        distance_description = f"with NPC's front {format_float(longitudinal_distance)} m in front of Ego's front"
    elif npc_data["longitudinal_distance"] < -0.2 and npc_data["longitudinal_distance"] > -4.8:
        front_back = "partially beside and slightly behind ego"
        longitudinal_distance = abs(npc_data["longitudinal_distance"])  
        distance_description = f"with NPC's front {format_float(longitudinal_distance)} m behind Ego's front"
    else:
        front_back = "longitudinally close but not overlapping with ego"
        distance_description = "still separated laterally"

    if npc_data["lateral_distance"] > 2.16: 
        left_right = "to the right of ego"
        lateral_distance = abs(npc_data["lateral_distance"]) - 2.16  
        lateral_description = f"with NPC's left {format_float(lateral_distance)} m to the right of Ego's right"
    elif npc_data["lateral_distance"] < -2.16:
        left_right = "to the left of ego"
        lateral_distance = abs(npc_data["lateral_distance"]) - 2.16  
        lateral_description = f"with NPC's right {format_float(lateral_distance)} m to the left of Ego's left"
    # elif npc_data["lateral_distance"] > -2.16 and npc_data["lateral_distance"] < 0: 
    elif npc_data["lateral_distance"] < 2.16 and npc_data["lateral_distance"] > 0.2: 
        left_right = "partially beside and slightly to the right of ego"
        lateral_distance = abs(npc_data["lateral_distance"])  
        lateral_description = f"with NPC's right {format_float(lateral_distance)} m to the right of Ego's side"
    elif npc_data["lateral_distance"] > -2.16 and npc_data["lateral_distance"] < -0.2:
        left_right = "partially beside and slightly to the left of ego"
        lateral_distance = abs(npc_data["lateral_distance"]) 
        lateral_description = f"with NPC's right {format_float(lateral_distance)} m to the right of Ego's side"
    else:
        left_right = "laterally aligned but not overlapping with ego"
        lateral_description = "still separated longitudinally"

    if npc_data['current_lane'] == ego_data['lane']:
        prompt = f"{npc_key} and ego are in the same lane."
        prompt += f"Laterally (horizontally), {npc_key} is {left_right},{lateral_description}, "
        prompt += f"and and longitudinally (vertically) {front_back}, {distance_description}, "
        prompt += f"traveling at a speed of {format_float(npc_data['speed'])} m/s."
    else:
        prompt = f"{npc_key} is in the {relative_side} {abs(npc_data['current_lane'] - ego_data['lane'])} lane. "
        prompt += f"Laterally (horizontally), {npc_key} is {left_right},{lateral_description}, "
        prompt += f"and and longitudinally (vertically) {front_back}, {distance_description}, "
        prompt += f"traveling at a speed of {format_float(npc_data['speed'])} m/s."

    return prompt


def generate_npc_prompt(npc_key, npc_data, ego_data):
    def format_float(val):
        return f"{val:.2f}"

    # Half size definitions
    EGO_HALF_LENGTH = 2.4
    NPC_HALF_LENGTH = 2.4
    EGO_HALF_WIDTH = 1.08
    NPC_HALF_WIDTH = 1.08

    # Longitudinal relation
    longitudinal_distance = npc_data["longitudinal_distance"]
    abs_long_dist = abs(longitudinal_distance)
    long_dir = "front of" if longitudinal_distance > 0 else "behind"

    # Lateral relation
    lateral_distance = npc_data["lateral_distance"]
    abs_lat_dist = abs(lateral_distance)
    lat_dir = "right" if lateral_distance > 0 else "left"

    prompt = f"{npc_key} is currently driving in 'lane {npc_data['current_lane']}. "
    prompt += f"It is approximately {format_float(abs_long_dist)} meters {long_dir} the ego vehicle longitudinally, "
    prompt += f"and about {format_float(abs_lat_dist)} meters to the {lat_dir} side laterally. "
    prompt += f"{npc_key} is traveling at a speed of {format_float(npc_data['speed'])} m/s. "
    # prompt += "No abrupt maneuvers have been observed in its recent motion."

    return prompt

def generate_ego_prompt_old(ego_data):
    return f"ego is in lane {ego_data['lane']}, traveling at a longitudinal speed of {abs(format_float(ego_data['self_longitudinal_speed']))} m/s and a lateral speed of {abs(format_float(ego_data['self_lateral_speed']))} m/s, moving straight ahead."

def generate_ego_prompt(ego_data):
    return f"ego is in lane {ego_data['lane']}, traveling at a speed of {format_float(ego_data['speed'])} m/s, moving straight ahead."

def generate_full_prompt(json_file):

    data = load_json_file(json_file)
    npc_list = []
    prompts = []
    prompt = generate_ego_prompt(data["ego_vehicle"])
    prompts.append(prompt)

    for npc_key, npc_data in data.items():
        if npc_key != "ego_vehicle" and  npc_data["longitudinal_distance"] > -1:
            npc_list.append(npc_key)
            prompt = generate_npc_prompt(npc_key, npc_data, data["ego_vehicle"])
            prompts.append(prompt)


    final_prompt = f"From the ego view, the lanes from left to right are numbered from -2, to -4, with a lane width of 3.5 meters. " + " ".join(prompts)
    # final_prompt_easy = f"In the BEV image, the lanes from left to right are numbered from -3 to -7, with a lane width of 3.5 meters. The ego vehicle is located in the center of the image." 
    # final_prompt_easy += f"Each vehicle has a width of 2.16 m and a length of 4.8 m, and the distances are measured from the each vehicles' center position." + " ".join(prompts)
    return final_prompt, npc_list

def generate_full_prompt_specific_npc(json_file, npc_list):

    data = load_json_file(json_file)
    prompts = []
    prompt = generate_ego_prompt(data["ego_vehicle"])
    prompts.append(prompt)

    for npc_key, npc_data in data.items():
        if npc_key != "ego_vehicle" and npc_key in npc_list:
            prompt = generate_npc_prompt(npc_key, npc_data, data["ego_vehicle"])
            prompts.append(prompt)

    final_prompt = f"From the ego view, the lanes from left to right are numbered from -2, to -4, with a lane width of 3.5 meters. " + " ".join(prompts)
    return final_prompt

def generate_vehicle_descriptions(base_path, current_frame, steps=[0, 5, 10, 15, 20]):
    descriptions = {}
    npc_info = None
    for step in steps:
        frame_id = current_frame - step
        time_offset = step * 0.1
        key = "vehicle_description" if step == 0 else f"vehicle_description_{time_offset:.1f}s"
        padded_id = f"{frame_id:04d}" 
        path = os.path.join(base_path, f"{padded_id}.json")
        if step == 0:
            vehicle_description, npc_info = generate_full_prompt(path)
        else:
            vehicle_description = generate_full_prompt_specific_npc(path, npc_info)
        descriptions[key] = vehicle_description
    return descriptions


def generate_supply_vehicle_descriptions(base_path, current_frame, npc_info, steps=[0, 5, 10, 15, 20]):
    descriptions = {}
    for step in steps:
        frame_id = current_frame - step
        time_offset = step * 0.1
        key = "vehicle_description" if step == 0 else f"vehicle_description_{time_offset:.1f}s"
        padded_id = f"{frame_id:04d}" 
        path = os.path.join(base_path, f"{padded_id}.json")
        vehicle_description = generate_full_prompt_specific_npc(path, npc_info)
        descriptions[key] = vehicle_description
    return descriptions
    


def genereta_supply_common_prompt(common_best_record):
    agent_prompt = ""
    unique_path = next(iter(common_best_record.values()))["path"]
    base_dir = os.path.dirname(unique_path)
    root_dir = os.path.dirname(base_dir)
    filename = os.path.basename(unique_path) 
    number_str = os.path.splitext(filename)[0]  
    number = int(number_str.lstrip('0')) 
    history_reason_dir = os.path.join(root_dir, 'result_graph/')
    history_reason_path = os.path.join(history_reason_dir, f"{number_str}.txt")
    reasons = extract_reasons(history_reason_path)
    front_left_dir = os.path.join(root_dir, 'rgb_front_left/')
    front_left_image = os.path.join(front_left_dir, f"{number_str}.png")
    front_dir = os.path.join(root_dir, 'rgb_front/')
    front_image = os.path.join(front_dir, f"{number_str}.png")
    front_right_dir = os.path.join(root_dir, 'rgb_front_right/')
    front_right_image = os.path.join(front_right_dir, f"{number_str}.png")
    dangerous_npcs = [
    data['dangerous_npc'].lower() 
    for data in common_best_record.values() 
    if 'dangerous_npc' in data]
    vehicle_descriptions = generate_supply_vehicle_descriptions(base_dir, number, dangerous_npcs)
    for agent, data in common_best_record.items():
        agent_prompt += (
    f"The behavior of NPC{agent} in the current scene and {data['dangerous_npc']} in this historical scene shares partial similarities, "
    "such as motion direction or relative position. However, they are not identical. "
    "Please analyze differences in distance, speed, heading, and context before making a judgment."
)
    
    front_left_image_encode = compress_and_encode(front_left_image)
    front_image_encode = compress_and_encode(front_image)
    front_right_image_encode = compress_and_encode(front_right_image)
    return vehicle_descriptions, agent_prompt, front_left_image_encode, front_image_encode, front_right_image_encode, reasons
                
def genereta_supply_vehicle_descriptions(agent, data):
    unique_path = data["path"]
    base_dir = os.path.dirname(unique_path)
    root_dir = os.path.dirname(base_dir)
    filename = os.path.basename(unique_path)  
    number_str = os.path.splitext(filename)[0] 
    number = int(number_str.lstrip('0'))  
    history_reason_dir = os.path.join(root_dir, 'result_graph/')
    history_reason_path = os.path.join(history_reason_dir, f"{number_str}.txt")
    reasons = extract_reasons(history_reason_path)
    front_left_dir = os.path.join(root_dir, 'rgb_front_left/')
    front_left_image = os.path.join(front_left_dir, f"{number_str}.png")
    front_dir = os.path.join(root_dir, 'rgb_front/')
    front_image = os.path.join(front_dir, f"{number_str}.png")
    front_right_dir = os.path.join(root_dir, 'rgb_front_right/')
    front_right_image = os.path.join(front_right_dir, f"{number_str}.png")
    dangerous_npcs = [
    data['dangerous_npc'].lower()]
    vehicle_descriptions = generate_supply_vehicle_descriptions(base_dir, number, dangerous_npcs)
    agent_prompt = (
    f"The behavior of NPC{agent} in the current scene and {data['dangerous_npc']} in this historical scene shares partial similarities, "
    "such as motion direction or relative position. However, they are not identical. "
    "Please analyze differences in distance, speed, heading, and context before making a judgment."
)

    front_left_image_encode = compress_and_encode(front_left_image)
    front_image_encode = compress_and_encode(front_image)
    front_right_image_encode = compress_and_encode(front_right_image)
    return vehicle_descriptions, agent_prompt, front_left_image_encode, front_image_encode, front_right_image_encode, reasons



def compress_and_encode(image_path, target_width=200):
    img = Image.open(image_path)
    w, h = img.size
    ratio = target_width / w
    target_height = int(h * ratio)

    # 向后兼容的 LANCZOS 采样
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

def parse_stream_response(completion, print_output=False):

    response_text = ""
    try:
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content_piece = getattr(delta, "content", None)
            if content_piece is not None:
                if print_output:
                    print(content_piece, end="", flush=True)
                response_text += content_piece
    except Exception as e:
        print(f"\n[error]: {e}")
    return response_text



def build_history_scene_fp(index, images, descriptions, final_risk_text):
    scene = [
        {"type": "text", "text": f"[Historical Scene #{index}]"},
        {"type": "text", "text": (
        "This is a historical scene where an NPC exhibited behavior that was previously mistaken as high-risk, "
        "leading to a false collision prediction. However, no collision actually occurred. "
        "While some behavioral patterns may be partially similar, use this scene only as a contextual reference. "
        "Do not assume the same outcome. Carefully judge the current scene based on actual trajectories and interactions."
         )},
    {"type": "text", "text": "Front-left camera image:"},
        {"type": "image_url", "image_url": {"url": images["front_left"]}},
        {"type": "text", "text": "Front camera image:"},
        {"type": "image_url", "image_url": {"url": images["front"]}},
        {"type": "text", "text": "Front-right camera image:"},
        {"type": "image_url", "image_url": {"url": images["front_right"]}},
    ]
    sorted_descriptions = []

    for key, desc in descriptions.items():
        if key == "vehicle_description":
            sorted_descriptions.append((0.0, "Scene description at t = 0.0s of this historical image", desc))
        else:
            seconds = float(key.replace("vehicle_description_", "").replace("s", ""))
            sorted_descriptions.append((seconds, f"Scene description from {seconds:.1f} seconds before the historical image", desc))

    sorted_descriptions.sort(key=lambda x: x[0])

    for _, label, desc in sorted_descriptions:
        scene.append({"type": "text", "text": f"{label}: {desc}"})
    return scene


def build_history_scene_tp(index, images, descriptions, final_risk_text):
    scene = [
        {"type": "text", "text": f"[Historical Scene #{index}]"},
        {"type": "text", "text": (
        "This is a historical scene where an NPC's behavior eventually led to a collision. "
        "While the scene may share some spatial or behavioral patterns with the current one, "
        "do not assume the same outcome will occur. Use this only as context. "
        "Your judgment should be based solely on the actual positions, speeds, and motion patterns in the current scene."
    )},
        {"type": "text", "text": "Front-left camera image:"},
        {"type": "image_url", "image_url": {"url": images["front_left"]}},
        {"type": "text", "text": "Front camera image:"},
        {"type": "image_url", "image_url": {"url": images["front"]}},
        {"type": "text", "text": "Front-right camera image:"},
        {"type": "image_url", "image_url": {"url": images["front_right"]}},
    ]
    sorted_descriptions = []

    for key, desc in descriptions.items():
        if key == "vehicle_description":
            sorted_descriptions.append((0.0, "Scene description at t = 0.0s of this historical image", desc))
        else:
            seconds = float(key.replace("vehicle_description_", "").replace("s", ""))
            sorted_descriptions.append((seconds, f"Scene description from {seconds:.1f} seconds before the historical image", desc))

    sorted_descriptions.sort(key=lambda x: x[0])

    for _, label, desc in sorted_descriptions:
        scene.append({"type": "text", "text": f"{label}: {desc}"})
    return scene

def get_collision_reasoning_with_image(base_dir, current_frame, current_scene, past_scene, \
                                       vehicle_description, vehicle_description_0_5s, vehicle_description_1s, vehicle_description_1_5s, vehicle_description_2s):
    client = OpenAI(
        base_url="https://yunwu.ai/v1",
        api_key='sk-Ko0JaD1xNhI3xJI7gCwUYdRTuCcoDnlJOUdZxS9vJxo33X9O',
        timeout=120)
    
    similar_result_tp = find_most_similar_graph_pair(current_scene, past_scene, db_path="/bdata/usrdata/root/monitor_test/Bench2Drive/graph_pair_db_collision.jsonl")
    similar_result_fp = find_most_similar_graph_pair(current_scene, past_scene, db_path="/bdata/usrdata/root/monitor_test/Bench2Drive/graph_pair_db.jsonl")
    # if similar_result_tp:
    #     similar_result_fp = find_most_similar_graph_pair_tp_graph(current_scene, past_scene, similar_result_tp, db_path="/bdata/usrdata/root/monitor_test/Bench2Drive/graph_pair_db.jsonl")
    # else:
    #     similar_result_fp = find_most_similar_graph_pair(current_scene, past_scene, db_path="/bdata/usrdata/root/monitor_test/Bench2Drive/graph_pair_db.jsonl")

    front_left_dir = os.path.join(base_dir, 'rgb_front_left/')
    id = f"{current_frame:04d}" 
    front_left_image = os.path.join(front_left_dir, f"{id}.png")
    front_dir = os.path.join(base_dir, 'rgb_front/')
    front_image = os.path.join(front_dir, f"{id}.png")
    front_right_dir = os.path.join(base_dir, 'rgb_front_right/')
    front_right_image = os.path.join(front_right_dir, f"{id}.png")
    front_left_img_b64_str = compress_and_encode(front_left_image)
    front_img_b64_str = compress_and_encode(front_image)
    front_right_img_b64_str = compress_and_encode(front_right_image)
    common_best_record_fp, unique_best_record_fp = None, None
    if similar_result_fp:
        common_best_record_fp, unique_best_record_fp = select_best_matches(similar_result_fp, front_left_image, front_image, front_right_image)
    common_vehicle_descriptions_fp, common_agent_prompt_fp  = None, None
    history_scenes_fp = []
    history_scenes_fp_file = {}
    index = 0
    if common_best_record_fp:
        common_vehicle_descriptions_fp, common_agent_prompt_fp, common_front_left_image_encode_fp, common_front_image_encode_fp, common_front_right_image_encode_fp, final_risk_text_fp = genereta_supply_common_prompt(common_best_record_fp)
        images = {
            "front_left": f"data:image/png;base64,{common_front_left_image_encode_fp}",
            "front": f"data:image/png;base64,{common_front_image_encode_fp}",
            "front_right": f"data:image/png;base64,{common_front_right_image_encode_fp}",
        }
        scene = build_history_scene_fp(index, images, common_vehicle_descriptions_fp, final_risk_text_fp)
        index += 1
        history_scenes_fp.extend(scene)
        history_scenes_fp_file["common"] = common_best_record_fp
    elif unique_best_record_fp:
        # few_shot_prompt = "The following are scenarios where have similar NPC behaviors with current scenario. Some of the NPC behaviors are thought to cause collisions, but they do not. Please refer to:\n"
        # best_agent_id, best_entry = max(unique_best_record_fp.items(), key=lambda item: item[1]["similarity_score"])
    
        # 只保留这一个
        # unique_best_record_fp = {best_agent_id: best_entry}
        for agent, data in unique_best_record_fp.items():
            # few_shot_prompt += f"the behavior of NPC{agent} in the current scene and {data['dangerous_npc']} in the past scene is similar,"
            vehicle_descriptions, common_agent_prompt_fp, front_left_image_encode, front_image_encode, front_right_image_encode, final_risk_text_tp = genereta_supply_vehicle_descriptions(agent, data)
            images = {
                "front_left": f"data:image/png;base64,{front_left_image_encode}",
                "front": f"data:image/png;base64,{front_image_encode}",
                "front_right": f"data:image/png;base64,{front_right_image_encode}",
            }
            scene = build_history_scene_fp(index, images, vehicle_descriptions, final_risk_text_tp)
            index += 1
            history_scenes_fp.extend(scene)
        history_scenes_fp_file["unique"] = unique_best_record_fp
    common_best_record_tp, unique_best_record_tp = None, None
    if similar_result_tp:
        common_best_record_tp, unique_best_record_tp = select_best_matches(similar_result_tp, front_left_image, front_image, front_right_image)
    common_vehicle_descriptions_tp, common_agent_prompt_tp  = None, None

    history_scenes_tp = []
    history_scenes_tp_file = {}
    if common_best_record_tp:
        common_vehicle_descriptions_tp, common_agent_prompt_tp, common_front_left_image_encode_tp, common_front_image_encode_tp, common_front_right_image_encode_tp, final_risk_text_tp = genereta_supply_common_prompt(common_best_record_tp)
        images = {
            "front_left": f"data:image/png;base64,{common_front_left_image_encode_tp}",
            "front": f"data:image/png;base64,{common_front_image_encode_tp}",
            "front_right": f"data:image/png;base64,{common_front_right_image_encode_tp}",
        }
        scene = build_history_scene_tp(index, images, common_vehicle_descriptions_tp, final_risk_text_tp)
        index += 1
        history_scenes_tp.extend(scene)
        history_scenes_tp_file["common"] = common_best_record_tp
    elif unique_best_record_tp:  
        # few_shot_prompt = "The following are scenarios where have similar NPC behaviors with current scenario. Some of the NPC behaviors are thought to cause collisions, but they do not. Please refer to:\n"
        # best_agent_id, best_entry = max(unique_best_record_tp.items(), key=lambda item: item[1]["similarity_score"])
    
        # # 只保留这一个
        # unique_best_record_tp = {best_agent_id: best_entry}
        for agent, data in unique_best_record_tp.items():
            # few_shot_prompt += f"the behavior of NPC{agent} in the current scene and {data['dangerous_npc']} in the past scene is similar,"
            vehicle_descriptions, common_agent_prompt_tp, front_left_image_encode, front_image_encode, front_right_image_encode, final_risk_text_tp = genereta_supply_vehicle_descriptions(agent, data)
            images = {
                "front_left": f"data:image/png;base64,{front_left_image_encode}",
                "front": f"data:image/png;base64,{front_image_encode}",
                "front_right": f"data:image/png;base64,{front_right_image_encode}",
            }
            scene = build_history_scene_tp(index, images, vehicle_descriptions, final_risk_text_tp)
            index += 1
            history_scenes_tp.extend(scene)
        history_scenes_tp_file["unique"] = unique_best_record_tp

    user_content = [
        {"type": "text", "text": f"[Current Scene needed to judge:]"},
        {"type": "text", "text": "This is the left front camera image that the autonomous vehicle perceives at the current moment"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_left_img_b64_str}"}},
        {"type": "text", "text": "This is the front camera image that the autonomous vehicle perceives at the current moment"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_img_b64_str}"}},
        {"type": "text", "text": "This is the right front camera image that the autonomous vehicle perceives at the current moment"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_right_img_b64_str}"}},
        {"type": "text", "text": f"This is pass 2s scene description: {vehicle_description_2s}"},
        {"type": "text", "text": f"This is pass 1.5s scene description: {vehicle_description_1_5s}"},
        {"type": "text", "text": f"This is pass 1s scene description: {vehicle_description_1s}"},
        {"type": "text", "text": f"This is pass 0.5s scene description: {vehicle_description_0_5s}"},
        {"type": "text", "text": f"This is current scene description: {vehicle_description}"},
    ]

    system_prompt = """
                You are an **autonomous vehicle collision prediction expert**. Your task is to determine whether a collision will occur **within the next 0.5 second**, based on:
                - **Three front-facing perspective images** (left-front, front, right-front), and**Motion descriptions for each visible vehicle (NPC) in the past 2 seconds**.
                
                Your focus is on **NPCs that are visible of the ego vehicle**, within the camera's field of view. Ignore vehicles that are outside the sensor's effective range.

                ### Risk Levels:
                - <Extreme Risk (0)>: Collision is imminent or extremely likely within **0.5 seconds**. Immediate avoidance required.
                - <Negligible Risk (1)>: No meaningful collision risk within **0.5 seconds**. NPC is moving away or remains at a safe distance.

                ---

                ### Step-by-Step Reasoning Process:

                1 **Identify NPCs in Ego’s Forward View**:
                - Use camera images and motion data to identify **all visible NPCs** ahead of the ego vehicle.
                - For each NPC, extract:
                - Lane position (same lane, adjacent lane, far lane)
                - Lateral and longitudinal distance relative to the ego
                - Whether the NPC is within, approaching, or crossing into the ego’s lane

                2 **Analyze NPC Motion Using Images + Past Descriptions**:

                ▶ From Images (Crucial for Lane Change Detection):

                - **Determine if the NPC is changing lanes or merging**:
                    - Look for visual signs:
                        - Is the vehicle angled relative to the lane?
                        - Are any wheels or parts of the body crossing lane markings?
                        - Is it straddling two lanes?
                    - Identify if the vehicle orientation suggests drift or turning
                    - Check lane alignment: centered, drifting, or encroaching

                - **Estimate relative positions**:
                    - Is the NPC close laterally (within 1.5 meters)?
                    - Is the vehicle longitudinally close (within 5–10 meters)?

                ▶ From Past Motion Descriptions:

                - Check:
                    - Acceleration or deceleration trend
                    - Lateral movement over time
                    - Whether the vehicle’s trajectory suggests merging into the ego’s path

                 Do not assume lane change based only on motion text. Use **visual evidence from images as primary source** for merge/lane-crossing behavior.

                3 **Predict Collision Risk**:
                - Use both image and motion data to answer the following hierarchical questions, each serving a specific purpose to reduce false positives and false negatives:

                    1.Is the NPC in or merging into ego’s lane?

                    2.Are lateral (<1.5 m) and longitudinal distances close enough to risk collision?

                    3.Is ego closing in on NPC (relative_speed > 0)?

                - If all above suggest threat, estimate TTC:

                    - relative_speed = ego_speed - npc_speed

                    - If relative_speed > 0: TTC = longitudinal_distance / relative_speed

                    - Else TTC = ∞

                - Interpret TTC and lane status with 3 risk levels:

                    - Extreme Risk (0): TTC < 0.5s and NPC in/merging lane, high confidence of collision

                    - Uncertain Risk (1): TTC near 0.5s or ambiguous merge/lane-change cues; collision possible but unclear

                    - Negligible Risk (2): TTC ≥ 0.7s, large gaps, NPC not merging, or ego not closing in

                 Do not judge collision risk by proximity alone; consider lateral/longitudinal distances and merge intent comprehensively.

                4 **Final Judgment and Action Plan**:
                - Will a collision occur? (Yes/No)
                - If **Yes**:
                - Which NPC(s) are dangerous?
                - What is the cause? (e.g., lane merging, too slow in front, sudden cut-in)
                - Recommended Action Plan: Recommend how the ego should react (e.g., brake, slow down, change lanes) and what speed range is safe.

                ### Output Format (JSON):
                Return the final output as a JSON with:

                ```json
                {
                    "risk_score": 0-2,
                    "reason": "Explain why this risk level was assigned, citing distances, speeds, and motion.",
                    "predict_npc_action": {
                        "NPC1": "braking",
                        "NPC2": "lane change left",
                        ...
                    },
                    "dangerous_npc: "npcs",
                    "avoidable_behaviors": {
                        "ego_action": ["brake", "lane change right"],
                        "ego_speed": "3.2 m/s"
                    }
                }                 """

    
    output_dir = os.path.join(base_dir, 'output_graph/')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{id}.txt")
    if len(history_scenes_fp) > 0 and len(history_scenes_tp) > 0:
        history_intro_fp = [
            # {"type": "text", "text": (
            #     "The following are historical reference scenes. Use them only as contrast to support or challenge your current judgment."
            # )}
            {"type": "text", "text": (
                """The following are historical reference scenes. Use them only as contrast to support or challenge your current judgment. Important: DO NOT copy conclusions from history scenes directly.
                    You must:
                    - Independently assess the current scene using current image + motion data.
                    - Then compare with each historical scene.
                    - Point out the **differences** in motion pattern, distance, or lane position.

                    Even if the past scene looks similar, small differences (like heading angle, relative speed, relative distance) can result in different risk outcomes.

                    Only use historical scenes as reference, NOT as ground truth."""
            )}
        ]
        scenes_fp = history_intro_fp + history_scenes_fp + [{"type": "text", "text": common_agent_prompt_fp}]

        history_intro_tp = [
            # {"type": "text", "text": (
            #     "The following are historical reference scenes. Use them only as contrast to support or challenge your current judgment."
            # )}
            {"type": "text", "text": (
                """The following are historical reference scenes. Use them only as contrast to support or challenge your current judgment. Important: DO NOT copy conclusions from history scenes directly.
                    You must:
                    - Independently assess the current scene using current image + motion data.
                    - Then compare with each historical scene.
                    - Point out the **differences** in motion pattern, distance, or lane position.

                    Even if the past scene looks similar, small differences (like heading angle, relative speed, relative distance) can result in different risk outcomes.

                    Only use historical scenes as reference, NOT as ground truth."""
            )}
        ]
        scenes_tp = history_intro_tp + history_scenes_tp + [{"type": "text", "text": common_agent_prompt_tp}]
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content + scenes_fp + scenes_tp}
        ] 
        filtered_history_scenes_fp = [item for item in scenes_fp if "image_url" not in item]
        filtered_history_scenes_tp = [item for item in scenes_tp if "image_url" not in item]
        filtered_user_content = [item for item in user_content if "image_url" not in item]
        combined_output = (
            [{"role": "user", "content": filtered_user_content}] +
            filtered_history_scenes_fp +
            filtered_history_scenes_tp +
            [{"type": "meta", "history_scenes_fp_file": history_scenes_tp_file}] +
            [{"type": "meta", "history_scenes_fp_file": history_scenes_fp_file}]
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_output, f, ensure_ascii=False, indent=2)

        # with open(output_file, "w", encoding="utf-8") as f:
        #     f.write(json.dumps([{"role": "user", "content": filtered_user_content}] + filtered_history_scenes_fp + filtered_history_scenes_tp, indent=2))

    elif len(history_scenes_fp) > 0:
        history_intro_fp = [
            {"type": "text", "text": (
                """The following are historical reference scenes. Use them only as contrast to support or challenge your current judgment. Important: DO NOT copy conclusions from history scenes directly.
                    You must:
                    - Independently assess the current scene using current image + motion data.
                    - Then compare with each historical scene.
                    - Point out the **differences** in motion pattern, distance, or lane position.

                    Even if the past scene looks similar, small differences (like heading angle, relative speed, relative distance) can result in different risk outcomes.

                    Only use historical scenes as reference, NOT as ground truth."""
            )}
        ]
        # print(type(history_intro_fp))
        # print(type(history_scenes_fp))
        # print(type(common_agent_prompt_fp))
        scenes_fp = history_intro_fp + history_scenes_fp + [{"type": "text", "text": common_agent_prompt_fp}]
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content + scenes_fp}
        ]
        # 写入文件（覆盖写入）
        filtered_history_scenes_fp = [item for item in scenes_fp if "image_url" not in item]
        filtered_user_content = [item for item in user_content if "image_url" not in item]

        combined_output = (
            [{"role": "user", "content": filtered_user_content}] +
            filtered_history_scenes_fp +
            [{"type": "meta", "history_scenes_fp_file": history_scenes_tp_file}] +
            [{"type": "meta", "history_scenes_fp_file": history_scenes_fp_file}]
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_output, f, ensure_ascii=False, indent=2)

        # with open(output_file, "w", encoding="utf-8") as f:
        #     f.write(json.dumps([{"role": "user", "content": filtered_user_content}] + filtered_history_scenes_fp, indent=2))  
    elif len(history_scenes_tp) > 0:
        history_intro_tp = [
            # {"type": "text", "text": (
            #     "The following are historical reference scenes. Use them only as contrast to support or challenge your current judgment."
            # )}
            {"type": "text", "text": (
                """The following are historical reference scenes. Use them only as contrast to support or challenge your current judgment. Important: DO NOT copy conclusions from history scenes directly.
                    You must:
                    - Independently assess the current scene using current image + motion data.
                    - Then compare with each historical scene.
                    - Point out the **differences** in motion pattern, distance, or lane position.

                    Even if the past scene looks similar, small differences (like heading angle, relative speed, relative distance) can result in different risk outcomes.

                    Only use historical scenes as reference, NOT as ground truth."""
            )}
        ]
        scenes_tp = history_intro_tp + history_scenes_tp + [{"type": "text", "text": common_agent_prompt_tp}]

        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content + scenes_tp}
        ]
        filtered_history_scenes_tp = [item for item in scenes_tp if "image_url" not in item]
        filtered_user_content = [item for item in user_content if "image_url" not in item]

        combined_output = (
            [{"role": "user", "content": filtered_user_content}] +
            filtered_history_scenes_tp +
            [{"type": "meta", "history_scenes_fp_file": history_scenes_tp_file}] +
            [{"type": "meta", "history_scenes_fp_file": history_scenes_fp_file}]
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_output, f, ensure_ascii=False, indent=2)
        # with open(output_file, "w", encoding="utf-8") as f:
        #     f.write(json.dumps([{"role": "user", "content": filtered_user_content}] + filtered_history_scenes_tp, indent=2))    
    else:
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
        ]
        filtered_user_content = [item for item in user_content if "image_url" not in item]
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps([{"role": "user", "content": filtered_user_content}], indent=2))  

    completion = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
        temperature=0.8,
        # response_format=CollisionPredict
    )
    # completion = client.chat.completions.create(
    #     # model="claude-3-5-sonnet-20240620",
    #     # model="qwen-omni-turbo-2025-01-19",
    #     messages=messages,
    #     temperature=0.8,
    #     # stream=True
    #     # # response_format=CollisionPredict
    # )
    
    print(completion)
    collision_reasoning = completion.choices[0].message.content
    return collision_reasoning
    # response_text = parse_stream_response(completion)

    # return response_text


