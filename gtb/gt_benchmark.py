import matplotlib.pyplot as plt
import numpy as np
import os
import json
import traceback
import time
import pandas as pd

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


from .utils import (
    map_to_list,
    extract_list,
    extract_dict,
    list_of_lists_to_string,
    find_character_position,
    overlap_dict,
    find_most_similar_images,
    )

from .gym_agent import CustomEnv, LLMAgent

from .fixers import pad_rows_to_max_length

def extract_slash(model_name):
    parts = model_name.split('/')
    return parts[1] if len(parts) > 1 else ''

def calculate_path_length(actions):
    # Initialize the agent's starting position
    x, y = 0, 0
    
    # Dictionary to map actions to their coordinate changes
    move_delta = {
        'move_up': (0, 1),
        'move_down': (0, -1),
        'move_left': (-1, 0),
        'move_right': (1, 0)
    }
    
    # Keep track of each position the agent moves to
    path_positions = [(x, y)]  # Start with the initial position
    path = []
    # Process each action
    for action in actions:
        if action in move_delta:
            dx, dy = move_delta[action]
            x += dx
            y += dy
            path_positions.append((x, y))  # Append new position after the move
            path.append({"x":dx,"y":dy})
    # Calculate the path length by summing the distances between consecutive positions
    path_length = 0
    for i in range(1, len(path_positions)):
        prev_x, prev_y = path_positions[i - 1]
        curr_x, curr_y = path_positions[i]
        path_length += abs(curr_x - prev_x) + abs(curr_y - prev_y)
    
    return path_length, path

def benchmark(model,
            total_episodes,
            character_discriptions_dict,
            world_map_fixed,
            world_map_fixed_with_chars,
            tileset_used_dict_1st_layer,
            tileset_used_dict,
            walkable_tiles_list,
            object_tiles_list,
            objective_tile_dict,
            astar_path_length):

    
    print("Generating Actions...")
    except_done = False
    whole_exception = 0
    
    frames = [] 
    episodes = 0
    #all_episodes_rewards = []
    #all_episode_path_lenght=[]
    #all_wrong_actions_generated=[] 
    #all_generation_errors=[]
    #all_total_achieved_objectives=[]
    #all_total_1tilewindow_achieved_objectives=[]
    #all_total_5tilewindow_achieved_objectives=[] 
    
    
    #all_episodes_rewards.append(1)

    """if "Meta" in model:
        model = AutoModelForCausalLM.from_pretrained(
                                    MODEL, 
                                    device_map="cuda", 
                                    torch_dtype="auto", 
                                    trust_remote_code=True, 
                                )
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

    if "Qwen" in MODEL:
        device = "cuda" # the device to load the model onto

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-4B-Chat",
            torch_dtype="auto",
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B-Chat")

    if "RWKV" in MODEL:
        model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(0)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, padding_side='left', pad_token="<s>")"""
    try:
        while not except_done:

            #folder_path = f"{save_dir}/{generation}/"
            folder_path = "/"
            
            #tile_images_1st_layer = load_image_dict(tileset_used_dict_1st_layer,folder_path)
            #tile_images = load_image_dict(tileset_used_dict,folder_path)
            
            """print(f"tile_images_1st_layer: {tileset_used_dict_1st_layer}")
            print(f"tile_images: {tileset_used_dict}")

            #tile_images_1st_layer, input_cost, output_cost = find_most_similar_images_gpt(tileset_used_dict_1st_layer,folder_path, openai.ChatCompletion.create, MODEL)
            #INPUT_TOKENS.append(input_cost)
            #OUTPUT_TOKENS.append(output_cost)
            #tile_images, input_cost, output_cost = find_most_similar_images_gpt(tileset_used_dict,folder_path, openai.ChatCompletion.create, MODEL)
            removed_value = tileset_used_dict.pop('Protagonist', None) 
            removed_value = tileset_used_dict.pop('Antagonist', None) 
            tileset_used_dict[character_discriptions_dict["Protagonist"]] = "@"
            tileset_used_dict[character_discriptions_dict["Antagonist"]] = "#"
            print("Retrieving images.")
            tile_images,_s= find_most_similar_images(tileset_used_dict,folder_path,no_check=True)
            print("Images Retrieved.")
            tile_images_1st_layer = overlap_dict(tile_images, tileset_used_dict_1st_layer)"""

            env = CustomEnv(walkable_tiles_list, "tile_images_1st_layer", "tile_images", world_map_fixed, world_map_fixed_with_chars, object_tiles_list, "#")
            
            agent = LLMAgent()
            state = env.reset()

            reward_feedback = "This is your first objective"
            reward_design = {
                "Each action you take will deduct following reward so that you take minimum amount of actions to complete objective. For example you take 10 actions then you will recieve -10 rewards": -1,
                "You are 8 tiles away from objective thus objective is incomplete": -100,
                "You are 5 to 8 tiles away from objective thus objective is incomplete": -50,
                "You are 3 to 5 tiles away from objective": +25,
                "You are 1 to 3 tiles away from objective": +50,
                "You are 1 tile away or your are on the objective tile from objective": +100,
                "You have completed the objective": +200,
            }
            
            done = False
            orig_world_map_fixed_with_chars = world_map_fixed_with_chars
            while not done:
                total_actions = {}
                prev_reward = 0
                all_rewards = 0
                all_llm_path_length = 0
                all_wrong_action_generated = 0
                all_total_actions_taken = 0
                all_generation_errors = 0
                all_total_achieved_objectives = 0
                all_total_1tilewindow_achieved_objectives = 0
                all_total_5tilewindow_achieved_objectives = 0
                total_llm_paths = []
                protagonist_position = find_character_position(orig_world_map_fixed_with_chars, "@")
                reward_this_objective = {}
                for i in range(len(objective_tile_dict)):
                    total_actions[list(objective_tile_dict.keys())[i]] = []
                    reward_this_objective[list(objective_tile_dict.keys())[i]] = []
                    for j in range(total_episodes):
                        
                        print("\n")
                        print(f"OBJECTIVE: {list(objective_tile_dict.keys())[i]}")
                        print(f"EPISODE: {j+1}")
                        print("\n")
                        reward = 0
                        llm_path_length = 0
                        total_actions_taken = 0
                        wrong_action_generated = 0
                        generation_error = 0
                        total_achieved_objectives = 0
                        total_1tilewindow_achieved_objectives = 0
                        total_5tilewindow_achieved_objectives = 0
                        llm_paths = []
                        #while not objective_flag:
                        action_system = f"You are a great planner in a 2D game. You plan actions for the protagonist of the game to achieve all objects. You are given objectives, tiles and the position of tiles to achieve the objectives. You have the following options as actions: 'move_up', move_down, 'move_right', and 'move_left'. Generate a sequence of actions that will achieve the objective. Only return the sequence of actions from the options."
                        #action_prompt = f"Given the story:\n{story}\n a 2D tile map of a world was created for the story:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\n and walkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Accumulative rewards for all the previous objectives tille now are {reward}. Taking this information into your context, create a sequence of actions for the protagonist to complete the objective: {list(objective_tile_dict.keys())[i]}, which is to reach the tile, 'pick_object' or 'hit_enemy' at tile and position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary. Do not return it in a Python response."
                        #action_prompt = f"Given the story:\n{story}\n a 2D tile map of a world was created for the story:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\n and walkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Accumulative rewards for all the previous objectives tille now are {reward}. Taking this information into your context, create a sequence of actions for the protagonist to complete the objective: {list(objective_tile_dict.keys())[i]}, which is to reach the tile, 'pick_object' or 'hit_enemy' at tile and position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary with one entry like 'action': [move_up, move_down.. etc.]. Do not return it in a Python response.
                        if j > 0:
                            if (distance_from_objective[0] == 0 and distance_from_objective[1] == 0):
                                reward = prev_reward
                                break
                            if i ==0:
                                action_prompt = f"Let's say you are given a 2D tile map of a 2D game:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\nwalkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {prev_protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. You are also given information about your previous try for this objective. You generated the following sequence of actions:\n{total_actions[list(objective_tile_dict.keys())[i]]}\n These actions took protagonist from coordinates {prev_protagonist_position} to {protagonist_position} which was {distance_from_objective} distance away from objective (the objective is at the tile and the position {list(objective_tile_dict.values())[i]}). This previous try gave you {reward_this_objective[list(objective_tile_dict.keys())[i]]} rewards. Taking this information into your context, create a sequence of actions for the agent to complete the objective which is to reach the tile, at the tile and the position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary with one entry like 'action': [move_up, move_down.. etc.]. Do not return it in a Python response."
                                protagonist_position = prev_protagonist_position
                            else:                                
                                action_prompt = f"Let's say you are given a 2D tile map of a 2D game:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\nwalkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {prev_protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. You are also given information about your previous try for all objectives. You generated the following sequence of actions:\n{total_actions[list(objective_tile_dict.keys())[i]]}\n These actions took protagonist from coordinates {prev_protagonist_position} to {protagonist_position} which was {distance_from_objective} distance away from objective (the objective is at the tile and the position {list(objective_tile_dict.values())[i]}). This previous try gave you {reward_this_objective[list(objective_tile_dict.keys())[i]]} rewards. Taking this information into your context, create a sequence of actions for the agent to complete the objective which is to reach the tile, at the tile and the position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary with one entry like 'action': [move_up, move_down.. etc.]. Do not return it in a Python response."
                                protagonist_position = prev_protagonist_position
                        else: 
                            if i ==0:
                                action_prompt = f"Let's say you are given a 2D tile map of a 2D game:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\nwalkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Taking this information into your context, create a sequence of actions for the agent to complete the objective which is to reach the tile, at the tile and the position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary with one entry like 'action': [move_up, move_down.. etc.]. Do not return it in a Python response."
                            else:
                                action_prompt = f"Let's say you are given a 2D tile map of a 2D game:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\nwalkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Taking this information into your context, create a sequence of actions for the agent to complete the objective which is to reach the tile, at the tile and the position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary with one entry like 'action': [move_up, move_down.. etc.]. Do not return it in a Python response."
                        #if i ==0:
                        #    action_prompt = f"Let's say you are given a 2D tile map of a 2D game:\n{world_map_fixed_with_chars}\n The tile map was created using 'A' and 'B', where 'A' is walkable and B is not walkable\n The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Taking this information into your context, create a sequence of actions for the agent to complete the objective which is to reach the tile, 'pick_object' or 'hit_enemy' at the tile and the position: 'A', {list(objective_tile_dict.values())[i][1:]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary with one entry like 'action': [move_up, move_down.. etc.]. Do not return it in a Python response."
                        #else:
                        #    action_prompt = f"Let's say you are given a 2D tile map of a 2D game:\n{world_map_fixed_with_chars}\n The tile map was created using 'A' and 'B', where 'A' is walkable and B is not walkable\n The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Accumulative rewards for all the previous objectives till now are {reward}. Taking this information into your context, create a sequence of actions for the agent to complete the objective which is to reach the tile, 'pick_object' or 'hit_enemy' at the tile and the position: 'A', {list(objective_tile_dict.values())[i][1:]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary with one entry like 'action': [move_up, move_down.. etc.]. Do not return it in a Python response."

                        action_exception = 0
                        action_done = False
                        while not action_done:
                            try:
                                if "gpt" in model:
                                    actions_discriptions = openai.ChatCompletion.create(model=model, messages=[
                                                                                                            {"role": "system", "content": action_system},
                                                                                                            {"role": "user", "content": action_prompt}
                                                                                                            ])
                                    
                                    #INPUT_TOKENS.append(actions_discriptions["usage"]["prompt_tokens"])
                                    #OUTPUT_TOKENS.append(actions_discriptions["usage"]["completion_tokens"])
                                    out_resp = actions_discriptions['choices'][0]['message']['content']
                                    print(f"response from model: {out_resp}")
                                    action_dict = extract_dict(out_resp)
                                """elif "claude" in MODEL:

                                    actions_discriptions = client.messages.create(model=MODEL, system=action_system,max_tokens=4096,messages=[{"role": "user", "content": action_prompt}])
                                    
                                    #INPUT_TOKENS.append(actions_discriptions["usage"]["prompt_tokens"])
                                    #OUTPUT_TOKENS.append(actions_discriptions["usage"]["completion_tokens"])
                                    action_dict = extract_dict(actions_discriptions.content[0].text)
                                elif "llama3" in MODEL:
                                    chat_completion = client_groq.chat.completions.create(
                                    #
                                    # Required parameters
                                    #
                                    messages=[
                                        # Set an optional system message. This sets the behavior of the
                                        # assistant and can be used to provide specific instructions for
                                        # how it should behave throughout the conversation.
                                        {
                                            "role": "system",
                                            "content": action_system
                                        },
                                        # Set a user message for the assistant to respond to.
                                        {
                                            "role": "user",
                                            "content": action_prompt,
                                        }
                                    ],

                                    # The language model which will generate the completion.
                                    model=MODEL,

                                    #
                                    # Optional parameters
                                    #

                                    # Controls randomness: lowering results in less random completions.
                                    # As the temperature approaches zero, the model will become deterministic
                                    # and repetitive.
                                    temperature=1.0,

                                    # The maximum number of tokens to generate. Requests can use up to
                                    # 32,768 tokens shared between prompt and completion.
                                    max_tokens=4096,
                                    )

                                    action_dict = extract_dict(chat_completion.choices[0].message.content)

                                elif "mixtral" in MODEL:
                                    chat_completion = client_groq.chat.completions.create(
                                    #
                                    # Required parameters
                                    #
                                    messages=[
                                        # Set an optional system message. This sets the behavior of the
                                        # assistant and can be used to provide specific instructions for
                                        # how it should behave throughout the conversation.
                                        {
                                            "role": "system",
                                            "content": action_system
                                        },
                                        # Set a user message for the assistant to respond to.
                                        {
                                            "role": "user",
                                            "content": action_prompt,
                                        }
                                    ],

                                    # The language model which will generate the completion.
                                    model=MODEL,

                                    #
                                    # Optional parameters
                                    #

                                    # Controls randomness: lowering results in less random completions.
                                    # As the temperature approaches zero, the model will become deterministic
                                    # and repetitive.
                                    temperature=1.0,

                                    # The maximum number of tokens to generate. Requests can use up to
                                    # 32,768 tokens shared between prompt and completion.
                                    max_tokens=4096,
                                    )

                                    action_dict = extract_dict(chat_completion.choices[0].message.content)

                                elif "gemma" in MODEL:
                                    chat_completion = client_groq.chat.completions.create(
                                    #
                                    # Required parameters
                                    #
                                    messages=[
                                        # Set an optional system message. This sets the behavior of the
                                        # assistant and can be used to provide specific instructions for
                                        # how it should behave throughout the conversation.
                                        {
                                            "role": "system",
                                            "content": action_system
                                        },
                                        # Set a user message for the assistant to respond to.
                                        {
                                            "role": "user",
                                            "content": action_prompt,
                                        }
                                    ],

                                    # The language model which will generate the completion.
                                    model=MODEL,

                                    #
                                    # Optional parameters
                                    #

                                    # Controls randomness: lowering results in less random completions.
                                    # As the temperature approaches zero, the model will become deterministic
                                    # and repetitive.
                                    temperature=1.0,

                                    # The maximum number of tokens to generate. Requests can use up to
                                    # 32,768 tokens shared between prompt and completion.
                                    max_tokens=4096,
                                    )

                                    action_dict = extract_dict(chat_completion.choices[0].message.content)

                                elif "Phi" in MODEL:

                                    model = AutoModelForCausalLM.from_pretrained(
                                        MODEL, 
                                        device_map="cuda", 
                                        torch_dtype="auto", 
                                        trust_remote_code=True, 
                                    )
                                    tokenizer = AutoTokenizer.from_pretrained(MODEL)

                                    messages = [
                                        {"role": "system", "content": action_system},
                                        {"role": "user", "content": action_prompt}
                                    ]

                                    pipe = pipeline(
                                        "text-generation",
                                        model=model,
                                        tokenizer=tokenizer,
                                    )

                                    generation_args = {
                                        "max_new_tokens": 500,
                                        "return_full_text": False,
                                        "temperature": 1.0,
                                        "do_sample": True,
                                    }

                                    output = pipe(messages, **generation_args)
                                    print(f"Action: {output[0]['generated_text']}")
                                    action_dict = extract_dict(output[0]['generated_text'])

                                elif "Meta" in MODEL:

                                    messages = [
                                        {"role": "system", "content": action_system},
                                        {"role": "user", "content": action_prompt}
                                    ]

                                    pipe = pipeline(
                                        "text-generation",
                                        model=model,
                                        tokenizer=tokenizer,
                                    )

                                    prompt = pipe.tokenizer.apply_chat_template(
                                            messages, 
                                            tokenize=False, 
                                            add_generation_prompt=True
                                    )

                                    terminators = [
                                        pipe.tokenizer.eos_token_id,
                                        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                                    ]

                                    outputs = pipe(
                                        prompt,
                                        max_new_tokens=256,
                                        eos_token_id=terminators,
                                        do_sample=True,
                                        temperature=1.0,
                                        #top_p=0.9,
                                    )
                                    
                                    print(outputs[0]["generated_text"][len(prompt):])
                                    action_dict = extract_dict(outputs[0]['generated_text'][len(action_prompt):])
                                    
                                    del prompt
                                    del outputs
                                    torch.cuda.empty_cache() 
                                    gc.collect()
                                    time.sleep(30)
                                
                                elif "Qwen" in MODEL:

                                    messages = [
                                        {"role": "system", "content": action_system},
                                        {"role": "user", "content": action_prompt}
                                    ]

                                    pipe = pipeline(
                                        "text-generation",
                                        model=model,
                                        tokenizer=tokenizer,
                                    )

                                    prompt = tokenizer.apply_chat_template(
                                            messages, 
                                            tokenize=False, 
                                            add_generation_prompt=True
                                    )

                                    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

                                    generated_ids = model.generate(
                                        model_inputs.input_ids,
                                        max_new_tokens=256
                                    )

                                    generated_ids = [
                                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                                    ]

                                    
                                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                                    
                                    print(f"Action: {response}")
                                    action_dict = extract_dict(response)
                                    
                                    del prompt
                                    del response
                                    torch.cuda.empty_cache() 
                                    gc.collect()
                                    time.sleep(30)
                                
                                """
                                
                                print("Action: \n")
                                print(action_dict["action"])
                                print("\n")
                                action_done = True

                            except Exception as e:
                                generation_error += 1
                                tb = traceback.format_exc()
                                print(f"Exception raised: {e}\n {tb}")
                                action_exception += 1
                                reward -= 1
                                reward_feedback = ""
                                reward_feedback = "Your previous objectives reward feedback is: "
                                reward_feedback += f"You are given a regret(negative reward) of -1 points an error that was a cause of wrong generation."
                                if action_exception >= 10:
                                    action_done = True
                                    reward -= astar_path_length
                                    reward -= 100
                                    reward_feedback += f"You were very far from the objective tile so you were also given a regret(negative reward) of -100 points and objective was INCOMPLETE"
                                    
                                continue
                        if action_exception >= 10:
                            action_done = True
                            continue
                        total_actions[list(objective_tile_dict.keys())[i]].append(action_dict["action"])
                        
                        _llm_path_length, llm_path = calculate_path_length(action_dict["action"])
                        llm_paths.append(llm_path)
                        llm_path_length += _llm_path_length
                        total_actions_taken += len(action_dict["action"])
                        
                        try:
                            for action_str in action_dict["action"]:
                                action = agent.action(action_str)
                                state, _r, done, _ = env.step(action)
                                
                                #frame = env.render(mode='rgb_array')  # Capture the frame
                                #frames.append(frame)  # Append the frame
                                time.sleep(0.01)
                        except Exception as e:
                            wrong_action_generated += 1
                            generation_error += 1
                            reward -= 1
                            reward_feedback = ""
                            reward_feedback = "Your previous objectives reward feedback is: "
                            reward_feedback += f"You are given a regret(negative reward) of -0.5 points an error that was a cause of wrong generation."
                            tb = traceback.format_exc()
                            print(f"Exception raised: {e}\n {tb}")
                    
                    
                        current_state = list_of_lists_to_string(state)
                        
                        print(current_state)
                        print("\n")
                        
                        #check_prompt = f"Given the previous world state:\n{world_map_fixed_with_chars}\n and the updated state that you returned: \n{current_state}\n is the objective {list(objective_tile_dict.keys())[i]} completed? Remember, from the dictionary of objectives, this objective will be completed when you reach tile {list(objective_tile_dict.values())[0]} at position {list(objective_tile_dict.values())[1]} or you are one tile aound this position in any directions. Strictly, only return 'Complete' or 'Incomplete'."
                        #
                        #check_discriptions = openai.ChatCompletion.create(model=MODEL, messages=[
                        #                                                                        {"role": "system", "content": action_system},
                        #                                                                        {"role": "user", "content": action_prompt},
                        #                                                                        {"role": "assistant", "content": actions_discriptions['choices'][0]['message']['content']},
                        #                                                                        {"role": "user", "content": check_prompt}
                        #                                                                        ])
                        
                        
                        world_map_fixed_with_chars = current_state
                        
                        #INPUT_TOKENS.append(check_discriptions["usage"]["prompt_tokens"])
                        #OUTPUT_TOKENS.append(check_discriptions["usage"]["completion_tokens"])
                        
                        for k, value in enumerate(objective_tile_dict.values()):
                            if k == i:
                                objective_pos = extract_list(str(value))
                        prev_protagonist_position = protagonist_position
                        protagonist_position = find_character_position(world_map_fixed_with_chars, "@")
                        print("\n")
                        print(f"protagonist_position: {protagonist_position}")
                        print(f"objective_position: [{objective_pos[1]},{objective_pos[2]}]")
                        

                        distance_from_objective = (abs(objective_pos[1] - protagonist_position[0]), abs(objective_pos[2] - protagonist_position[1]))
                        print(f"distance from current objective: [{distance_from_objective[0]}, {distance_from_objective[1]}]") 
                        print("\n")

                        reward_feedback = ""
                        reward_feedback = "Your previous objectives reward feedback is: "
                        reward -= len(action_dict["action"])
                        reward_feedback += f"You took {len(action_dict['action'])} actions for the objective so you were given a regret(negative reward) of -{len(action_dict['action'])} points. "
                        if (distance_from_objective[0] > 8 or distance_from_objective[1] > 8):
                            reward -= 100
                            reward_feedback += f"You were very far from the objective tile so you were given a regret(negative reward) of -100 points and objective was INCOMPLETE"
                        if (distance_from_objective[0] > 5 and distance_from_objective[0] < 8) or (distance_from_objective[1] > 5 and distance_from_objective[1] < 8):
                            reward -= 50
                            reward_feedback += f"You were far from the objective tile so you were given a regret(negative reward) of -50 points and objective was INCOMPLETE"
                        if (distance_from_objective[0] <= 5 and distance_from_objective[0] > 3) and (distance_from_objective[1] <= 5 and distance_from_objective[1] > 3):
                            reward += 25
                            reward_feedback += f"You were close to the objective tile so you were given a reward of 25 points"
                        if (distance_from_objective[0] < 3 and distance_from_objective[0] > 1) and (distance_from_objective[1] < 3 and distance_from_objective[1] > 1):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"

                        if (distance_from_objective[0] <= 1) and (distance_from_objective[1] > 1 and distance_from_objective[1] <= 5):
                            total_5tilewindow_achieved_objectives += 1
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"
                        if (distance_from_objective[1] <= 1) and (distance_from_objective[0] > 1 and distance_from_objective[0] <= 5):
                            total_5tilewindow_achieved_objectives += 1
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"

                        if (distance_from_objective[0] <= 1 and distance_from_objective[1] <= 1):# or check_discriptions['choices'][0]['message']['content'] == "Complete":
                            
                            if (distance_from_objective[0] == 0 and distance_from_objective[1] == 0):# and check_discriptions['choices'][0]['message']['content'] == "Complete":
                                reward += 200
                                total_achieved_objectives += 1
                                #objective_flag = True
                                reward_feedback += f"You were by the objective tile and you COMPLETED the objective so you were given a reward of 200 points"
                            else:
                                total_1tilewindow_achieved_objectives += 1
                                reward += 100
                                reward_feedback += f"You were by the objective tile so you were given a reward of 100 points"
                        
                        reward_this_objective[list(objective_tile_dict.keys())[i]].append(reward)
                        print("\n")
                        print(f"EPISODE REWARDS uptill now: {reward}")
                        print("\n")
                        prev_reward = reward
                    all_rewards += reward
                    all_llm_path_length += llm_path_length
                    all_wrong_action_generated += wrong_action_generated
                    all_generation_errors += generation_error
                    all_total_achieved_objectives += total_achieved_objectives
                    all_total_1tilewindow_achieved_objectives += total_1tilewindow_achieved_objectives
                    all_total_5tilewindow_achieved_objectives += total_5tilewindow_achieved_objectives
                    all_total_actions_taken += total_actions_taken
                    total_llm_paths.append(llm_paths)
                    print("\n")
                    print(f"All REWARDS uptill now: {all_rewards}")
                    print("\n")



                
                #all_episodes_rewards.append(all_rewards)
                #best_reward_index = np.argmax(all_episodes_rewards)
                #all_episode_path_lenght.append(llm_path_length)
                #all_wrong_actions_generated.append(wrong_action_generated)
                #all_generation_errors.append(generation_error)
                #all_total_achieved_objectives.append(total_achieved_objectives)
                #all_total_1tilewindow_achieved_objectives.append(total_1tilewindow_achieved_objectives)
                #all_total_5tilewindow_achieved_objectives.append(total_5tilewindow_achieved_objectives)
                print("\n")
                print(f"TOTAL REWARD for EPISODE: {all_rewards}")
                episodes += 1
                
                done = True

            #with imageio.get_writer(f'C:/Users/DELL/Projects/Research/Story-to-Game/story-to-game/outputs/benchmark/{EXPERIMENT}_{generation}.mp4', fps=10) as video:
            #    for frame in frames:
            #        video.append_data(frame)

            except_done = True
        
    except Exception as e:
        #print(f"check#3 done = {done}")
        tb = traceback.format_exc()
        print(f"Exception raised: {e}\n {tb}")
        whole_exception += 1
        #if whole_exception >= 5:
        except_done = True
        pass
    

    total_possible_rewards = (len(objective_tile_dict)*200) - astar_path_length
    total_normalised_rewards = all_rewards/(total_possible_rewards)
    
    return all_rewards, total_possible_rewards, total_normalised_rewards, all_llm_path_length, all_total_actions_taken, total_llm_paths, all_wrong_action_generated, all_generation_errors, all_total_achieved_objectives, all_total_1tilewindow_achieved_objectives, all_total_5tilewindow_achieved_objectives

def save_data(benchmark_data, save_dir, file_name, save_json=False, save_csv=False):
    # Convert dictionary to DataFrame
    if "/" in file_name:
        file_name = extract_slash(file_name)

    if save_csv:
        df = pd.DataFrame([benchmark_data])
        # Append to CSV file
        if os.path.exists(f'{save_dir}/{file_name}.csv'):
            df.to_csv(f'{save_dir}/{file_name}.csv', mode='a', header=False, index=False)
        else:
            df.to_csv(f'{save_dir}/{file_name}.csv', mode='w', header=True, index=False)
    if save_json:
        # Append to JSON file
        if os.path.exists(f'{save_dir}/{file_name}.json'):
            with open(f'{save_dir}/{file_name}.json', 'r') as file:
                data_list = json.load(file)
        else:
            data_list = []
        # Append new data
        data_list.append(benchmark_data)
        # Write back to JSON file
        with open(f'{save_dir}/{file_name}.json', 'w') as file:
            json.dump(data_list, file, indent=4)


def run(model: str, total_episodes: int = 1, experiment_name: str = "exp_001", save_dir: str = "/outputs"):
    #save_dir = f"C:/Users/DELL/Projects/Research/Story-to-Game/story-to-game/outputs/benchmark"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'./data/traversal_benchmark.json', 'r') as file:
        bench_data = json.load(file)

    for i, data in enumerate(bench_data[:2]):
        try:
            print(f"EVALUATING ROW {i}")
            print(f"WITH EXPERIMENT ID {data['experiment_id']}")
            character_discriptions_dict = {}
            gen_story = data["story"]
            str_world = data["environment"]
            print(f"World:\n{str_world}\n")
            char_tile_mapping = data["tile_mapping"]
            walkables = data["walkable_tiles"]
            interactive_object_tiles = data["interactive_object_tiles"]
            
            
            character_discriptions_dict["Protagonist"] =  "green-cloaked archer"
            character_discriptions_dict["Antagonist"] =  "dark-robed sorceress"
            objective_tile_dict = data["objectives"]
            astar_path_length = data["path_length"]
            str_world = pad_rows_to_max_length(str_world)
            grid_world = map_to_list(str_world)
            world_width = max(len(row) for row in grid_world)
            world_height = len(grid_world)
            print(f"World dimensions: {world_width} x {world_height}")
            rewards, total_possible_rewards, normalised_rewards, llm_path_length, total_actions_taken, llm_path, wrong_action_generated, \
            generation_errors, total_achieved_objectives, \
            total_1tilewindow_achieved_objectives, total_5tilewindow_achieved_objectives = benchmark(model,total_episodes,character_discriptions_dict,str_world,str_world,char_tile_mapping,char_tile_mapping,
                                                                                                    walkables,interactive_object_tiles,objective_tile_dict,astar_path_length)

            model_results = {}

            model_results = {"experiment_id": data["experiment_id"],
                            "agent_rewards": rewards,
                            "total_possible_rewards": total_possible_rewards,
                            "normalised_agent_rewards":normalised_rewards,
                            "llm_path_length": llm_path_length,
                            "total_actions_taken": total_actions_taken,
                            "wrong_action_generated": wrong_action_generated,
                            "generation_errors": generation_errors,
                            "total_achieved_objectives": total_achieved_objectives,
                            "total_1tilewindow_achieved_objectives": total_1tilewindow_achieved_objectives,
                            "total_5tilewindow_achieved_objectives": total_5tilewindow_achieved_objectives,
                            "Path": llm_path
                            }
            
            print(f"RESULT FOR EXPERIMENT ID {data['experiment_id']}:")
            for key, val in model_results.items():
                print(f"{key} : {val}")
            exp = "003"    
            save_data(benchmark_data=model_results,file_name=f"{model}_results_{experiment_name}",save_dir=save_dir,save_json=True)
                
            
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Exception raised: {e}\n {tb}")

def logcosh(x):
    # s always has real part >= 0
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)

def compile_results(model: str, experiment_name: str):
    #file_name = "benchmark_results_final"
    file_name = f"RESULTS_Prelim_{experiment_name}"
    res_dir = f"C:/Users/DELL/Projects/Research/Story-to-Game/story-to-game/outputs/benchmark"
    
    if "/" in model:
        model = extract_slash(model)
    else:
        model = model
    with open(f'{res_dir}/{model}_results_{experiment_name}.json', 'r') as file:
        model_results = json.load(file)
    with open(f'{res_dir}/traversal_benchmark_w_binary.json', 'r') as file:
        bench_results = json.load(file)
    max_objectives = []
    astar_path_length = []
    for benchmark_data in bench_results:
        max_objectives.append(len(benchmark_data["objectives"]))
        astar_path_length.append(benchmark_data["path_length"])
    print("="*20)
    print("DATA STATS:\n")
    print(f"Maean number of objectives: {np.mean(max_objectives)}")
    print(f"Max number of objectives: {max(max_objectives)}")
    print(f"Min number of objectives: {min(max_objectives)}")
    print(f"Mean AStar Path Length: {np.mean(astar_path_length)}")
    print(f"Min AStar Path Length: {min(astar_path_length)}")
    print(f"Max AStar Path Length: {max(astar_path_length)}")
    print()
    print("="*20)
    total_normalised_scaled_rewards = []
    total_path_length_mse = []
    total_path_length_logcosh = []
    total_wrong_action_generated = []
    total_generation_errors = []
    total_possible_rewards_sum = []
    agent_rewards_total = []
    total_achieved_objectives_all = []
    total_1tilewindow_achieved_objectives_all = []
    total_5tilewindow_achieved_objectives_all = []
    all_objectives = []
    reward_scale_min = 0
    reward_scale_max = 100
    for i, data in enumerate(model_results):
       
        for benchmark_data in bench_results:
            if data["experiment_id"] == benchmark_data["experiment_id"]:

                no_of_objectives = len(benchmark_data["objectives"])
                norm_reward = (data["normalised_agent_rewards"] * (no_of_objectives*200 - benchmark_data["path_length"]))
                min_old = -(no_of_objectives*100 +  benchmark_data["path_length"])
                max_old = no_of_objectives*200 -  benchmark_data["path_length"]
                
                norm_scaled_reward = ((norm_reward - min_old)/(max_old - min_old))*(reward_scale_max-reward_scale_min)
                total_normalised_scaled_rewards.append(norm_scaled_reward)
                
                total_path_length_mse.append((benchmark_data["path_length"] - data["llm_path_length"]) ** 2)
                total_path_length_logcosh.append(logcosh(data["llm_path_length"] - benchmark_data["path_length"]))
                total_wrong_action_generated.append(data["wrong_action_generated"])
                total_generation_errors.append(data["generation_errors"])
                total_possible_rewards_sum.append(data["total_possible_rewards"])
                agent_rewards_total.append(data["agent_rewards"])
                total_achieved_objectives_all.append(data["total_achieved_objectives"])
                total_1tilewindow_achieved_objectives_all.append(data["total_1tilewindow_achieved_objectives"])
                total_5tilewindow_achieved_objectives_all.append(data["total_5tilewindow_achieved_objectives"])
                all_objectives.append(no_of_objectives)
    completion = (len(model_results)/len(bench_results))*100
    total_agent_rewards = sum(total_normalised_scaled_rewards)/len(total_normalised_scaled_rewards)
    path_len_mse = sum(total_path_length_mse)/len(total_path_length_mse)
    path_len_rmse = np.sqrt(sum(total_path_length_mse)/len(total_path_length_mse))
    path_len_logcosh = sum(total_path_length_logcosh)
    error_generation = sum(total_generation_errors)/len(total_generation_errors)
    action_errors = sum(total_wrong_action_generated)/len(total_wrong_action_generated)
    mean_agent_rewards = np.mean(agent_rewards_total)
    mean_total_possible_rewards = np.mean(total_possible_rewards_sum)
    percent_total_achieved_objective = sum(total_achieved_objectives_all)/(sum(all_objectives))*100
    #mean_1tilewindow_total_achieved_objective = np.mean(total_1tilewindow_achieved_objectives_all)
    percent_1tilewindow_total_achieved_objective = sum(total_1tilewindow_achieved_objectives_all)/(sum(all_objectives))*100
    percent_5tilewindow_total_achieved_objective = sum(total_5tilewindow_achieved_objectives_all)/(sum(all_objectives))*100
    #completion = 100
    #error_generation = 3
    total_score = (total_agent_rewards)# - (0.5*(error_generation))
    print(f"Results for model: {model}")
    print(f"total solved: {len(total_normalised_scaled_rewards)} out of {len(bench_results)}")
    print(f"Total normalised rewards: {total_agent_rewards}")
    print(f"Mean agent rewards {mean_agent_rewards}")
    print(f"Mean total possible rewards {mean_total_possible_rewards}")
    print(f"Path length MSE: {path_len_mse}")
    print(f"Path length RMSE: {path_len_rmse}")
    print(f"Path length LogCosh: {path_len_logcosh}")
    print(f"Total Generation Errors: {sum(total_generation_errors)}")
    print(f"Mean Error Generation: {error_generation}")
    print(f"Mean wrong actions generated: {action_errors}")
    print(f"Percent Total Achieved Objectives: {percent_total_achieved_objective}")
    print(f"Percent Total 5 Tile Window Achieved Objectives: {percent_5tilewindow_total_achieved_objective}")
    print("="*20)
    print(f"TOTAL SCORE: {total_score}")
    print("="*20)
    final_model_results = {}
    final_model_results = {
        "Model": model,
        "Total_Normalised_Agent_Rewards": total_agent_rewards,
        "mean_agent_rewards": mean_agent_rewards,
        "mean_total_possible_rewards":mean_total_possible_rewards,
        "Path_Len_MSE": path_len_mse,
        "Path_Len_RMSE": path_len_rmse,
        "Path_Len_LogCosh": path_len_logcosh,
        "Mean Generation Error": error_generation,
        "Mean Action Error": action_errors,
        "total_achieved_objectives_all": sum(total_achieved_objectives_all),
        "Percent Total Achieved Objectives": percent_total_achieved_objective,
        "total_1tilewindow_achieved_objectives_all": sum(total_1tilewindow_achieved_objectives_all),
        "total_5tilewindow_achieved_objectives_all": sum(total_5tilewindow_achieved_objectives_all),
        "Percent Total 1 tile window Achieved Objectives": percent_1tilewindow_total_achieved_objective,
        "Percent Total 5 tile window Achieved Objectives": percent_5tilewindow_total_achieved_objective,
        "all_objectives": sum(all_objectives),
        "Total_Score": total_score
    }
    save_data(benchmark_data=final_model_results,file_name=file_name,save_dir=res_dir,save_json=True)

