from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def response():
    # Make sure the model path is correct for your system!
    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    callback = CallbackManager( [StreamingStdOutCallbackHandler()] )
    llm = LlamaCpp(
        model_path="models/orca-2-7b.Q4_K_M.gguf",
        # n_gpu_layers=n_gpu_layers,
        temperature=0.5,
        top_p=1,
        callback_manager=callback,
        verbose=True,  # Verbose is required to pass to the callback manager
        # n_batch=n_batch,

    )
    question = "Who is the CEO of EIStudy?"
    
    info = """Members of Ei's Management Team 
    Pranav Kothari - Chief Executive Officer 
    Pranav Kothari is the current CEO of Educational Initiatives (Ei). He has been with Ei for over a decade and has made significant contributions to the company through his involvement in Ei Mindspark (the personalised and adaptive learning solution), and as the head of HR, Gifted Student Education, and Ei Shiksha. Under his leadership, Ei Mindspark has been deployed and recognized as the only software-based EdTech tool that has demonstrated a significant learning impact as independently measured by third parties like J-PAL. 
    """

    template = """You are a cautious reason Ai chatbot and follow the instructions.
    Be polite, precise and brief.
    you will be asked a question, understand the question and go through complete information.
    focus only on the relevant information.
    Then answer the question in less than 50 words.

    Question: {question}
    Refer the below information to answer, Do not use any other knowledge base:
    {info}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["question","info"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    resp = llm_chain.run({'question':question, 'info':info})
    return resp

results = response()

print(results)
