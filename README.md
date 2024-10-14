# Game-Traversal-Benchmark

This repository contains the _GameTraversalBenchmark (GTB)_, which is explained in the paper [GameTraversalBenchmark: Evaluating Planning Abilities Of Large Language Models Through Traversing 2D Game Maps](https://arxiv.org/abs/2410.07765).


### Usage

Install via repository:

```
git clone https://github.com/umair-nasir14/Game-Traversal-Benchmark.git GTB
cd GTB
```

If you want to replicate experiments in the paper or you want to evaluate you LLM on the benchmark then install via `environment.yml`:

```
conda env create -f environment.yml
conda activate gtbench
```

If you want to explore and get the data only:

```
pip install -r requirements.txt
```

To get data, simply:

```
from gtb.data import get_data

benchmark_data = get_data()

for i, data in enumerate(benchmark_data[:1]):
    print(data["environment"])
```

Also, `save_data()` is provided to save your results.

To replicate results in the paper, create a `.env` file and add the API keys:

```
OPENAI_API_KEY="sk..."
GROQ_API_KEY= ...
```

and install relevent library, such as `pip install openai`

```
from gtb.gtbench import run 
from gtb.evaluations import compile_results

model_name = "o1-preview-2024-09-12"
experiment_name = run(model_name)
compile_results(model_name, experiment_name)
```

This will give you a model specific result file with the name as `{model}_{experiment_name}.json` that will contain results for each row of data. There will be another file that will have all results combined to have `GTB_Score` and other scores. 

Models tested in the paper:

```
 "o1-preview-2024-09-12"
 "o1-mini-2024-09-12"
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

## Leaderboard:

| **Model**               | **GTBS(&uarr;)** | **MGE(&darr;)** | **MPL(&darr;)** | **MAT(&darr;)** | **Top-0 Acc.(&uarr;)** | **Top-1 Acc.(&uarr;)** | **Top-5 Acc.(&uarr;)** |
|-------------------------|--------------------|---------------------|---------------------|---------------------|--------------------------|--------------------------|--------------------------|
| **O1**                  | $67.84$   | $0.12$     | $51.35$    | $51.73$    | $50$            | $10.76$         | $13.19$         |
| **O1-mini**                  | $61.36$   | $0.83$     | $82.83$    | $82.95$    | $46.98$            | $6.70$         | $14.38$         |
| **GPT-4-Turbo**         | $44.97 \pm 0.22$   | $0.03 \pm 0.01$     | $80.91 \pm 0.69$    | $80.97 \pm 0.62$    | $19.2\pm0.24$            | $17.66 \pm 0.46$         | $23.05 \pm 1.03$         |
| **GPT-4-o**             | $30.95 \pm 0.76$   | $0.53 \pm 0.06$     | $85.42 \pm 0.58$    | $85.91 \pm 0.61$    | $7.84 \pm 0.17$          | $11.34 \pm 0.36$         | $18.99 \pm 0.45$         |
| **Claude-3-Opus**       | $28.65 \pm 0.59$   | $0.02 \pm 0.01$     | $100.41 \pm 0.37$   | $100.44 \pm 0.36$   | $5.49 \pm 0.65$          | $12.35 \pm 0.43$         | $22.72 \pm 0.09$         |
| **Claude-3-Sonnet**     | $18.54 \pm 0.22$   | $0.0 \pm 0.0$       | $75.05 \pm 0.38$    | $76.22 \pm 0.41$    | $0.73 \pm 0.05$          | $3.80 \pm 0.11$          | $13.64 \pm 0.31$         |
| **Random-FP**           | $18.02 \pm 0.50$   | N/A                 | N/A                 | N/A                 | $0.91 \pm 0.13$          | $2.77 \pm 0.09$          | $12.41 \pm 0.27$         |
| **Gemma-7B**            | $15.65 \pm 0.21$   | $0.11 \pm 0.02$     | $37.06 \pm 0.55$    | $40.16 \pm 0.71$    | $0.29 \pm 0.05$          | $2.05 \pm 0.05$          | $10.95 \pm 1.41$         |
| **GPT-3.5-Turbo**       | $14.34 \pm 0.31$   | $16.49 \pm 0.39$    | $46.83 \pm 0.69$    | $48.23 \pm 0.91$    | $0.44 \pm 0.23$          | $3.49 \pm 0.34$          | $11.88 \pm 0.31$         |
| **LLaMa-3-8B**          | $14.08 \pm 0.67$   | $0.54 \pm 0.06$     | $55.38 \pm 1.19$    | $56.06 \pm 1.12$    | $0.21 \pm 0.00$          | $2.27 \pm 0.32$          | $8.54 \pm 0.53$          |
| **LLaMa-3-70B**         | $11.39 \pm 1.36$   | $0.41 \pm 0.04$     | $266.88 \pm 23.79$  | $267.02 \pm 23.88$  | $1.06 \pm 0.05$          | $4.84 \pm 0.7$           | $16.63 \pm 0.96$         |
| **Claude-3-Haiku**      | $10.81 \pm 1.15$   | $0.0 \pm 0.0$       | $69.14 \pm 8.18$    | $70.09 \pm 8.21$    | $0.09 \pm 0.06$          | $1.25 \pm 0.88$          | $7.32 \pm 1.96$          |
| **Mixtral-8x7B** | $9.35 \pm 0.56$    | $9.61 \pm 0.41$     | $152.73 \pm 5.43$   | $152.99 \pm 5.46$   | $0.67 \pm 0.24$          | $2.85 \pm 0.16$          | $10.19 \pm 0.19$         |
| **Random-RP**           | $3.04 \pm 0.65$    | N/A                 | $278.48 \pm 0.67$   | $297.22 \pm 1.02$   | $0.0 \pm 0.0$            | $0.20 \pm 0.18$          | $15.37 \pm 2.26$         |


### Cite:

```
@article{nasir2024gametraversalbenchmark,
  title={GameTraversalBenchmark: Evaluating Planning Abilities Of Large Language Models Through Traversing 2D Game Maps},
  author={Nasir, Muhammad Umair and James, Steven and Togelius, Julian},
  journal={arXiv preprint arXiv:2410.07765},
  year={2024}
}
```
