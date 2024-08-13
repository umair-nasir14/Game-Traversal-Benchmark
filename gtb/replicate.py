import matplotlib.pyplot as plt
import numpy as np
import os
import json
import traceback
import time
import pandas as pd

from .data import get_data, save_data

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


from .utils import (
    map_to_list,
    extract_list,
    extract_dict,
    list_of_lists_to_string,
    find_character_position,
    extract_slash
    )

from .gym_agent import CustomEnv, LLMAgent

from .fixers import pad_rows_to_max_length

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
            world_map_fixed,
            world_map_fixed_with_chars,
            tileset_used_dict,
            walkable_tiles_list,
            object_tiles_list,
            objective_tile_dict,
            astar_path_length,
            client):

    
    print("Generating Actions...")
    except_done = False
    whole_exception = 0
    
    frames = [] 
    episodes = 0


    try:
        while not except_done:

            folder_path = "/"


            env = CustomEnv(walkable_tiles_list, world_map_fixed, world_map_fixed_with_chars, object_tiles_list, "#")
            
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
                    
                        action_system = f"You are a great planner in a 2D game. You plan actions for the protagonist of the game to achieve all objects. You are given objectives, tiles and the position of tiles to achieve the objectives. You have the following options as actions: 'move_up', move_down, 'move_right', and 'move_left'. Generate a sequence of actions that will achieve the objective. Only return the sequence of actions from the options."
                        
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
                    
                        action_exception = 0
                        action_done = False
                        while not action_done:
                            try:
                                if "gpt" in model:
                                    actions_discriptions = client.chat.completions.create(model=model, messages=[
                                                                                                            {"role": "system", "content": action_system},
                                                                                                            {"role": "user", "content": action_prompt}
                                                                                                            ])
                                    
                        
                                    out_resp = actions_discriptions['choices'][0]['message']['content']
                                    print(f"response from model: {out_resp}")
                                    action_dict = extract_dict(out_resp)
                                elif "claude" in model:

                                    actions_discriptions = client.messages.create(model=model, system=action_system,max_tokens=4096,messages=[{"role": "user", "content": action_prompt}])
                                    
                                    #INPUT_TOKENS.append(actions_discriptions["usage"]["prompt_tokens"])
                                    #OUTPUT_TOKENS.append(actions_discriptions["usage"]["completion_tokens"])
                                    action_dict = extract_dict(actions_discriptions.content[0].text)
                                elif "llama3" in model:
                                    chat_completion = client.chat.completions.create(
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
                                    model=model,

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

                                elif "mixtral" in model:
                                    chat_completion = client.chat.completions.create(
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
                                    model=model,

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

                                elif "gemma" in model:
                                    chat_completion = client.chat.completions.create(
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
                                    model=model,

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

                        world_map_fixed_with_chars = current_state

                        
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

                print("\n")
                print(f"TOTAL REWARD for EPISODE: {all_rewards}")
                episodes += 1
                
                done = True

            #with imageio.get_writer(f'./outputs/benchmark/{EXPERIMENT}_{generation}.mp4', fps=10) as video:
            #    for frame in frames:
            #        video.append_data(frame)

            except_done = True
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Exception raised: {e}\n {tb}")
        whole_exception += 1
        except_done = True
        pass
    

    total_possible_rewards = (len(objective_tile_dict)*200) - astar_path_length
    total_normalised_rewards = all_rewards/(total_possible_rewards)
    
    return all_rewards, total_possible_rewards, total_normalised_rewards, all_llm_path_length, all_total_actions_taken, total_llm_paths, all_wrong_action_generated, all_generation_errors, all_total_achieved_objectives, all_total_1tilewindow_achieved_objectives, all_total_5tilewindow_achieved_objectives


def run(model: str, total_episodes: int = 1, experiment_name: str = "exp_001", save_dir: str = "./outputs/"):

    if "gpt" in model:

        from openai import OpenAI
        client = OpenAI(os.getenv("OPENAI_API_KEY"))

    elif "claude" in model:

        import anthropic

        client = anthropic.Anthropic(
        os.environ.get("ANTHROPIC_API_KEY")
        
        )

    elif ("llama3" or "gemma" or "mixtral") in model:
        
        from groq import Groq

        client = Groq(
            api_key="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",
        )



    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bench_data = get_data()

    for i, data in enumerate(bench_data[:1]):
        try:
            print(f"EVALUATING ROW {i}")
            print(f"WITH EXPERIMENT ID {data['experiment_id']}")

            str_world = data["environment"]
            print(f"World:\n{str_world}\n")
            char_tile_mapping = data["tile_mapping"]
            walkables = data["walkable_tiles"]
            interactive_object_tiles = data["interactive_object_tiles"]
            
            objective_tile_dict = data["objectives"]
            astar_path_length = data["path_length"]
            str_world = pad_rows_to_max_length(str_world)
            grid_world = map_to_list(str_world)
            world_width = max(len(row) for row in grid_world)
            world_height = len(grid_world)
            print(f"Game dimensions: {world_width} x {world_height}")
            rewards, total_possible_rewards, normalised_rewards, llm_path_length, total_actions_taken, llm_path, wrong_action_generated, \
            generation_errors, total_achieved_objectives, \
            total_1tilewindow_achieved_objectives, total_5tilewindow_achieved_objectives = benchmark(model,total_episodes,str_world,str_world,char_tile_mapping,
                                                                                                    walkables,interactive_object_tiles,objective_tile_dict,astar_path_length, client)

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
            save_data(benchmark_data=model_results,file_name=f"{model}_{experiment_name}",save_dir=save_dir,save_json=True)
                
            
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Exception raised: {e}\n {tb}")