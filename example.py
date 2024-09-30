from gtb.gtbench import run 
from gtb.evaluations import compile_results

model_name = "o1-preview-2024-09-12"
experiment_name = run(model_name)
compile_results(model_name, experiment_name)