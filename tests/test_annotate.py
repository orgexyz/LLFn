import os

import pytest
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from llfn import LLFn


@pytest.fixture
def chat_model():
    return ChatOpenAI(
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
    )  # type: ignore


@pytest.fixture
def llm():
    return OpenAI(
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-davinci-003",
    )  # type: ignore


function_prompt = LLFn()


@function_prompt
def predict_animal(text: str) -> str:
    return f"What animal fits the following description the best: {text}"


predict_animal.expect("It has four legs and barks")("obviously a dog")
predict_animal.expect("It has four legs and meows")("obviously a cat")


def test_predict_animal_annotate_chat_model(chat_model):
    predict_animal.bind(chat_model)
    assert predict_animal("It has two legs and flies") == "obviously a bird"


def test_predict_animal_annotate_llm(llm):
    predict_animal.bind(llm)
    assert predict_animal("It has two legs and flies") == "obviously a bird"
