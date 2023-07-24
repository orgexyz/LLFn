---
title: Core Concept
---

### ⭐️ Defining Your Function
The primary goal of LLFn is to encapsulate everything you do with LLMs into a function. Each function comprises of 3 components: `input`, `prompt`, and `output`. Here's how you can create a simple text translator using LLFn.

```python
from llfn import LLFn

function_prompt = LLFn()

@function_prompt
def translate(text: str, output_language: str) -> str:
    return f"Translate the following text into {output_language} language: {text}"
```

### 🥪 Binding It with LLM

To execute this you have to `bind` the function to an LLM. You can bind any [LangChain's supported language models](https://python.langchain.com/docs/modules/model_io/models/) to LLFn functions.
```python
from langchain.chat_models import ChatOpenAI

# Setup LLM and bind it to the function
llm = ChatOpenAI(temperature=0.7, openai_api_key=API_KEY)
translate.bind(llm)
```

If you don't want to repeat yourself binding all the functions individually, you can bind directly to `function_prompt` as well.

```python
# Alternatively: bind the llm to all functions
function_prompt.bind(llm)
```

### 🍺 Calling the Function
Now you can call your function like you would for any Python function.

```
translate("Hello welcome. How are you?", "Thai")
```

The beauty of this construct is that you're able to infinitely compose your applications. LLFn does not make any assumption on how you'd chain up your application logic. The only requirement for a `function_prompt` is that it returns a prompt string at the end.

And ... that's it! That's just about everything you need to learn about LLFn to start building AI apps with it.

<details><summary><b>👆 Click to see the full code</b></summary>
<p>

```python
import os
from llfn import LLFn
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.7,
    model=os.getenv("OPENAI_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

function_prompt = LLFn()
function_prompt.bind(llm)

@function_prompt
def translate(text: str, output_language: str) -> str:
    return f"Translate the following text into {output_language} language: {text}"

# Call the function
print(translate("Hello welcome. How are you?", "Thai"))
```

</p>
</details>