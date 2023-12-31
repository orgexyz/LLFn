import os
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from llfn import LLFn


function_prompt = LLFn()


llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
)  # type: ignore


class AnimalResult(BaseModel):
    animal: str = Field(..., description="The animal that fits the question")


@function_prompt(AnimalResult)
def predict_animal(text: str):
    return f"What animal fits the following description the best: {text}"


predict_animal.expect("It has four legs and barks")(AnimalResult(animal="dog"))
predict_animal.expect("It has four legs and meows")(AnimalResult(animal="cat"))

function_prompt.bind(llm)
print(predict_animal("It has two legs and flies"))  # obviously a bird
