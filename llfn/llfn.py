import inspect
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Type, TypeVar, Union, Any
from functools import update_wrapper

from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field


class _ExampleTranslationResult(BaseModel):
    result_fr: str = Field(..., description="Translation result in French")
    result_es: str = Field(..., description="Translation result in Spanish")
    result_cn: str = Field(..., description="Translation result in Chinese")


def result_type(result_type: type) -> Type[BaseModel]:
    class Result(BaseModel):
        result: result_type = Field(..., description="The result of the instruction")

        def _is_llfn_result(self):
            return True

    return Result


T = TypeVar("T", bound=BaseModel)


@dataclass
class Example(Generic[T]):
    prompt: str
    result: T


BASIC_EXAMPLE_1 = Example(
    prompt='Please translate "dog" into different languages',
    result=_ExampleTranslationResult(
        result_fr="chien",
        result_es="perro",
        result_cn="狗",
    ),
)

BASIC_EXAMPLE_2 = Example(
    prompt="What animal has 4 legs and can bark?",
    result=result_type(str)(result="dog"),
)


def _run_completion(
    llm: Union[BaseChatModel, BaseLLM], messages: List[BaseMessage]
) -> str:
    """
    Run a completion on the given LLM with the given messages.

    Args:
        llm: The LLM to run the completion on. Must be either a ChatModel or a BaseLLM.
        messages: The messages to run the completion on.

    Returns:
        The completion result as a string.
    """

    if isinstance(llm, BaseChatModel):
        return llm.predict_messages(messages).content
    elif isinstance(llm, BaseLLM):
        texts = []
        for message in messages:
            if message.type == "system":
                texts.append(f"System: {message.content}")
            elif message.type == "human":
                texts.append(f"User: {message.content}")
            elif message.type == "ai":
                texts.append(f"Assistant: {message.content}")
            else:
                raise ValueError(f"Unknown message type: {message.type}")
        texts.append("Assistant: ")
        return llm("\n\n".join(texts))
    else:
        raise ValueError(f"LLM must be either a ChatModel or a BaseLLM; got {llm}")


class LLFnFunc:
    """
    This class is a wrapper around a user-provided function that is used to generate
    a prompt. LLFnFunc can be invoked as a function, which in turn invokes the
    user-provided function and passes the result to the LLM. The result will be parsed
    and return back to the caller.
    """

    app: "LLFn"
    func: Callable[..., str]
    llm: Optional[BaseLLM]
    examples: List[Example]
    result_type: Type[BaseModel]

    def __init__(
        self,
        app: "LLFn",
        func: Callable[..., str],
        result_type: Type[BaseModel],
    ):
        """
        Initizalize a new LLFnFunc instance. This is called by the LLFn class.

        Args:
            app: The LLFn instance that this function is bound to
            func: The user-provided function that generates a prompt
            return_type: The type of the return value to parse the LLM output into
        """
        self.app = app
        self.func = func
        self.llm = None
        self.examples = []
        self.result_type = result_type
        update_wrapper(self, func)

    def bind(self, llm):
        self.llm = llm

    def expect(self, *args, **kwargs):
        prompt = self.func(*args, **kwargs)

        def wrapper(result):
            if not isinstance(result, BaseModel):
                result = self.result_type(result=result)
            self.examples.append(Example(prompt=prompt, result=result))

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
        system_prompt = """
- You MUST process the user's command and produce exactly one result without any other contexts or explanations
- User command always begins with a Pydantic schema. Your output MUST be a JSON dump of a Pydantic object of that schema
- Output format is EXTREMELY important. Make sure it's a valid JSON dump of the Pydantic object.
"""
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for example in self.examples or [BASIC_EXAMPLE_1, BASIC_EXAMPLE_2]:
            messages.append(
                HumanMessage(
                    content=f"{type(example.result).schema()}\n{example.prompt}"
                )
            )
            messages.append(AIMessage(content=example.result.json()))
        messages.append(
            HumanMessage(content=f"{self.result_type.schema()}\n{user_prompt}")
        )
        output = _run_completion(llm, messages)
        result = self.result_type.parse_raw(output)
        if hasattr(result, "_is_llfn_result"):
            result = result.result
        return result


class LLFn:
    llm: Optional[Union[BaseChatModel, BaseLLM]]

    def __init__(self, llm: Optional[Union[BaseChatModel, BaseLLM]] = None):
        self.llm = llm

    def bind(self, llm: Union[BaseChatModel, BaseLLM]):
        self.llm = llm

    def __call__(self, arg: Union[type, Callable]) -> Any:
        return_type: Type[BaseModel]

        def wrap(func: Callable):
            return LLFnFunc(self, func, return_type)

        if inspect.isfunction(arg):
            r = arg.__annotations__["return"]
            return_type = result_type(r) if not isinstance(arg, BaseModel) else r
            return wrap(arg)
        if isinstance(arg, type):
            r = arg
            return_type = result_type(r) if not isinstance(arg, BaseModel) else r
            return wrap
        raise ValueError(f"Invalid argument type: {type(arg)}")
