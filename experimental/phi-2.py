import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/orca-2-7b.Q4_K_M.gguf"

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("cuda")
else:
    torch.set_default_device("cpu")
    print("cpu")

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, trust_remote_code=True)

inputs = tokenizer('''
    Instruct: """You are a cautious reason Ai chatbot and follow the instructions.
    Be polite, precise and brief.
    you will generate the code and evaluate them

    Question: Generate a code for adding two matrixs
    """
    Output:''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=400)
text = tokenizer.batch_decode(outputs)[0]
print(text)