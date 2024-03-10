from ctransformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import pipeline


# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    model_path_or_repo_id="models/orca-2-7b.Q4_K_M.gguf",
    model_type="llama",
    local_files_only=True,
    gpu_layers=50,
)
print(llm.config)
print(llm.embed("India"))
print(llm("which continent is france located?"))