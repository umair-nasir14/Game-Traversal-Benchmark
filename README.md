# Game-Traversal-Benchmark

This repository contains the _Game-Traversal-Benchmark (GTB)_, which is explained in the paper "add paper link".


### Usage

Install via repository:

```
git clone https://github.com/umair-nasir14/Game-Traversal-Benchmark.git GTB
cd GTB
```

To get data, simply:

```
from gtb.data import get_data


benchmark_data = get_data()

for i, data in enumerate(benchmark_data[:1]):
    print(data["environment"])
```

To replicate results in the paper, create a `.env` file and add the API keys:

```
OPENAI_API_KEY="sk..."
GROQ_API_KEY= ...
```

and install relevent library, such as `pip install openai`

```
from gtb.replicate import run 

model = "gpt-4-turbo-2024-04-09"
experiment_name = "replicate_001"
run(model = model, experiment_name = experiment_name)
compile_results(model = model, experiment_name = experiment_name)
```

This will give you a model specific result file with the name as `{model}_{experiment_name}.json` that will contain results for each row of data. There will be another file that will have all results combined to have `GTB_Score` and other scores. 

Models tested in the paper:

```
 "gpt-3.5-turbo-0125"
 "gpt-4-0613"
 "gpt-4-turbo-2024-04-09"
 "gpt-4o-2024-05-13"
 "claude-3-opus-20240229"
 "claude-3-sonnet-20240229"
 "claude-3-haiku-20240307"
 "llama3-8b-8192"
 "llama3-70b-8192"
 "mixtral-8x7b-32768"
 "gemma-7b-it"
```

### Cite:

```
Citation
```
