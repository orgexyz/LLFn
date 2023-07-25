import os
from langchain.chat_models import ChatOpenAI
from llfn import LLFn


function_prompt = LLFn()


llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
)  # type: ignore


@function_prompt
def predict_animal(text: str) -> str:
    return f"What animal fits the following description the best: {text}"


predict_animal.expect("It has four legs and barks")("obviously a dog")
predict_animal.expect("It has four legs and meows")("obviously a cat")

function_prompt.bind(llm)
print(predict_animal("It has two legs and flies"))  # obviously a bird
