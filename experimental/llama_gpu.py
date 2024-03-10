from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from llama_cpp import Llama

# model_path = "models/orca-2-7b.Q4_K_M.gguf"
model_path = "WizardCoder/mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf"
# model_path="models/zephyr/zephyr-7b-beta.Q4_K_M.gguf"

def response():
    # Make sure the model path is correct for your system!
    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    callback = CallbackManager( [StreamingStdOutCallbackHandler()] )
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.5,
        top_p=1,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        # max_tokens=2000,
        # n_ctx=2048,
        n_predict=50,
        callback_manager=callback,
        verbose=True,  # Verbose is required to pass to the callback manager
        
        # n_batch=n_batch,
    )

    # llm = Llama(
    #     model_path=model_path, 
    #     n_gpu_layers=28,
    #     n_threads=6,
    #     n_ctx=3584,
    #     n_batch=521,
    #     verbose=True
    #     ), 

    question = " write a tail recursion program in java"
    
    info = """
In this tail recursive version, the actual calculation is done in the factorialTailRec 
method with an additional parameter (accumulator) to accumulate the partial result. 
The recursive call is the last operation, making it tail recursive. The factorial method
serves as a wrapper for the initial call, passing the initial accumulator value.
 """

    template = """
    Question: {question}
    Refer the below information to answer:
    {info}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["question","info"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    resp = llm_chain.invoke({'question':question, 'info':info})
    return resp

results = response()

print(results)
