import numpy as np
import json

from .utils import extract_slash
from .data import save_data

def logcosh(x):
    # s always has real part >= 0
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)


def compile_results(model: str, experiment_name: str):
    #file_name = "benchmark_results_final"
    file_name = f"RESULTS_{experiment_name}"
    res_dir = "outputs"
    
    if "/" in model:
        model = extract_slash(model)
    else:
        model = model
    with open(f'{res_dir}/{experiment_name}.json', 'r') as file:
        model_results = json.load(file)
    with open('data/traversal_benchmark.json', 'r') as file:
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
    total_path_length = []
    total_actions_taken = []
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

                total_path_length.append(data["llm_path_length"])
                total_actions_taken.append(data["total_actions_taken"])
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
    total_score = sum(total_normalised_scaled_rewards)/len(total_normalised_scaled_rewards)
    mean_path_length = sum(total_path_length)/len(total_path_length)
    mean_actions_taken = sum(total_actions_taken)/len(total_actions_taken)
    path_len_mse = sum(total_path_length_mse)/len(total_path_length_mse)
    path_len_rmse = np.sqrt(sum(total_path_length_mse)/len(total_path_length_mse))
    path_len_logcosh = sum(total_path_length_logcosh)
    error_generation = sum(total_generation_errors)/len(total_generation_errors)
    action_errors = sum(total_wrong_action_generated)/len(total_wrong_action_generated)
    mean_agent_rewards = np.mean(agent_rewards_total)
    mean_total_possible_rewards = np.mean(total_possible_rewards_sum)
    percent_total_achieved_objective = sum(total_achieved_objectives_all)/(sum(all_objectives))*100
    percent_1tilewindow_total_achieved_objective = sum(total_1tilewindow_achieved_objectives_all)/(sum(all_objectives))*100
    percent_5tilewindow_total_achieved_objective = sum(total_5tilewindow_achieved_objectives_all)/(sum(all_objectives))*100

    print(f"Results for model: {model}")
    print(f"total solved: {len(total_normalised_scaled_rewards)} out of {len(bench_results)}")
    print(f"Total normalised rewards: {total_score}")
    print(f"Mean agent rewards {mean_agent_rewards}")
    print(f"Mean total possible rewards {mean_total_possible_rewards}")
    print(f"Mean Path Length: {mean_path_length}")
    print(f"Mean Actions Taken: {mean_actions_taken}")
    print(f"Total Generation Errors: {sum(total_generation_errors)}")
    print(f"Mean Error Generation: {error_generation}")
    print(f"Mean wrong actions generated: {action_errors}")
    print(f"Percent Total Achieved Objectives: {percent_total_achieved_objective}")
    print(f"Percent Total 5 Tile Window Achieved Objectives: {percent_5tilewindow_total_achieved_objective}")
    print("="*20)
    print(f"GTB_Score: {total_score}")
    print("="*20)
    final_model_results = {}
    final_model_results = {
        "Model": model,
        "Total_Normalised_Agent_Rewards": total_score,
        "mean_agent_rewards": mean_agent_rewards,
        "mean_total_possible_rewards":mean_total_possible_rewards,
        "mean_path_length": mean_path_length,
        "mean_actions_taken": mean_actions_taken,
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
        "GTB_Score": total_score
    }
    save_data(benchmark_data=final_model_results,file_name=file_name,save_dir=res_dir,save_json=True)

