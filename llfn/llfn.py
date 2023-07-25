from functools import update_wrapper
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List


class LLFnFunc:
    def __init__(self, app, func):
        self.app = app
        self.func = func
        self.llm = None
        self.examples = []

        class Result(BaseModel):
            result: func.__annotations__["return"] = Field(
                ..., description="The result of the instruction"
            )

        self.result_type = Result
        update_wrapper(self, func)

    def bind(self, llm):
        self.llm = llm

    def expect(self, *args, **kwargs):
        def wrapper(inner_result):
            prompt = self.func(*args, **kwargs)
            expected_result = self.result_type(result=inner_result)
            self.examples.append((prompt, expected_result.json()))

        return wrapper

    def __call__(self, *args, **kwargs):
        if self.llm is not None:
            llm = self.llm
        elif self.app.llm is not None:
            llm = self.app.llm
        else:
            raise ValueError(
                f'You must call "bind" before calling "{self.func.__name__}"'
            )
        user_prompt = self.func(*args, **kwargs)
        system_prompt = f"""
- You MUST process the user's command and produce exactly one result without any other contexts or explanations
- Your output must be a JSON dump of a Pydantic object of schema: {self.result_type.schema()}
- Output format is EXTREMELY important. Make sure it's a valid JSON dump of the Pydantic object.
"""
        if isinstance(llm, BaseChatModel):
            messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
            for prompt, expected_result in self.examples:
                messages.append(
                    HumanMessage(
                        content=f"""
Example command: {prompt}\n\n
{expected_result}"""
                    )
                )
            messages.append(HumanMessage(content=user_prompt))
            output = llm.predict_messages(messages).content
        elif isinstance(llm, BaseLLM):
            message = f"{system_prompt}\n\n"
            for prompt, expected_result in self.examples:
                message += f"""
Example command: {prompt}\n\n
{expected_result}\n\n"""
            message += f"{user_prompt}"
            output = llm(message)
        else:
            raise ValueError(f"LLM must be either a ChatModel or a BaseLLM")
        return self.result_type.parse_raw(output).result


class LLFn:
    def __init__(self):
        self.llm = None

    def bind(self, llm):
        self.llm = llm

    def __call__(self, func):
        return LLFnFunc(self, func)
