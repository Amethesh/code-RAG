from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

from .backend.controller.code_generator import code_generator
from .backend.controller.logic_highlighter_gpt import get_gpt_response
from .backend.controller.code_remover import remove_content
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"hello": "world"}


@app.post("/getGenerateCode")
def generate_code(requestData: Dict):
    question = requestData.get("question")
    language = requestData.get("language")

    fullCode = code_generator(question, language)
    highlightCode = get_gpt_response(fullCode)
    cutCode = remove_content(fullCode, highlightCode)

    print(cutCode)
    return cutCode


@app.post("/getAnalysisCode")
def analysis_code(resquestData: Dict):
    userCode = resquestData.get("userCode")
    # code_analysis(userCode)


if __name__ == "__main__":
    uvicorn.run(app)
