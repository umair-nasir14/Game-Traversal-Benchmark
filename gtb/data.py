import json



def get_data():
    
    with open(f'./data/traversal_benchmark.json', 'r') as file:
        bench_data = json.load(file)

    return bench_data