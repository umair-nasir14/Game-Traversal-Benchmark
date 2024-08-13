import json
import pandas as pd
import os

from utils import extract_slash


def get_data():
    
    with open(f'./data/traversal_benchmark.json', 'r') as file:
        bench_data = json.load(file)

    return bench_data


def save_data(benchmark_data, save_dir, file_name, save_json=False, save_csv=False):
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