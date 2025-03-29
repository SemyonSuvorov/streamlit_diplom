from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="models-cache", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="models-cache", trust_remote_code=True)

print(model)