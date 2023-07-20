import os
from langchain.chat_models import ChatOpenAI
from llfn import LLFn
from pydantic import BaseModel, Field
from typing import List


function_prompt = LLFn()


class AgentLogicOutput(BaseModel):
    tasks: List[str] = Field(
        ..., description="list of tasks to accomplish the given goal"
    )
    reasoning: str = Field(..., description="why you choose to do these tasks")


@function_prompt
def agent_think(goal: str) -> AgentLogicOutput:
    return f"You're an agent planning 5 tasks to accomplish the following goal: {goal}"


llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL"),
)  # type: ignore


function_prompt.bind(llm)
print(agent_think("Creating online business"))
