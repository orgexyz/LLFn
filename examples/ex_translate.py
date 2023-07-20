import os
from langchain.chat_models import ChatOpenAI
from llfn import LLFn


function_prompt = LLFn()


@function_prompt
def translate(text: str, output_language: str) -> str:
    return f"Translate the following text into {output_language} language: {text}"


llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
)  # type: ignore

function_prompt.bind(llm)
print(translate("Hello welcome. How are you?", "Thai"))
