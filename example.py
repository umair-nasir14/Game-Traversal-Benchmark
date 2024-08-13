from gtb.replicate import run 
from gtb.evaluations import compile_results

run("llama3-8b-8192")
compile_results("llama3-8b-8192", "exp_001")