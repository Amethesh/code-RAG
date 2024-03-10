from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()
# MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH")
MODEL_PATH = "generation_models/mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf"


# def code_generator(question, info, language):
def code_generator(question, language):
    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = (
        512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    )

    callback = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        top_p=1,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_predict=50,
        callback_manager=callback,
        verbose=True,
        # max_tokens=2000,
        # n_ctx=2048,
        # n_batch=n_batch,
    )

    # question = "write a tail recursion program"

    info = """
    """

    template = f"""
    Question: {question}
    generate ONLY the code in {language}:
    
    
    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["question", "info"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    code = llm_chain.invoke({"question": question, "info": info})
    return code


results = code_generator(
    "Write a program for multipling two number from user? ", "JAVA"
)

print(results)
