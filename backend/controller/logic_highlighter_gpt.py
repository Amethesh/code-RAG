from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_gpt_response():

    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        verbose=True,
        streaming=True,
    )

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
    ONLY Tell me the line in which there is the main logic in the code
    
    """

    prompt = PromptTemplate(template=template, input_variables=["code"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    code = llm_chain.invoke({"code": code})
    return code


# Get the GPT-3 response
results = get_gpt_response()

# Print the result in a formatted manner
print("GPT-3 Response:")
print("----------------")
print("Code:")
print(results["code"])
print("\nExplanation:")
print(results["text"])
