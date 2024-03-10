from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH="generation_models/orca-2-7b.Q4_K_M.gguf"
# MODEL_PATH="generation_models/mistral-7b-instruct-v0.2-code-ft.Q4_K_M.gguf"

def code_generator():
    n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

    callback = CallbackManager( [StreamingStdOutCallbackHandler()] )
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.2,
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
    
    code = """
import java.util.Scanner;

public class MultiplyNumbers {
    public static void main(String[] args) {
        
        Scanner scanner = new Scanner(System.in);

        
        System.out.print("Enter the first number: ");
        
        double num1 = scanner.nextDouble();

        
        System.out.print("Enter the second number: ");
        
        double num2 = scanner.nextDouble();


        scanner.close();

        
        double result = num1 * num2;

        
        System.out.println("The product of " + num1 + " and " + num2 + " is: " + result);
    }
}

    """

    template = """
    {code}
    Tell me the line in which there is the main logic in the code
    
    """

    prompt = PromptTemplate(template=template, input_variables=["code"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    code = llm_chain.invoke({'code':code})
    return code

results = code_generator()

print(results)
