<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://github.com/orgexyz/LLFn/assets/891585/521df7cc-2675-41ba-ac5d-8ca57f92261b" alt="Logo" height="100">
  </a>
  <p align="center">
    <b>LLFn (read: 🐘 Elephant)</b> is a light-weight framework for creating applications using LLMs.
    <br />
    Build anything from a simple text-summarizer to complex AI agents with one common primitive: <b>function</b>.
    <br />
    <div align="center">
        <a href="https://github.com/orgexyz/LLFn/graphs/contributors">
          <img src="https://img.shields.io/github/contributors/orgexyz/LLFn.svg?style=for-the-badge" />
        </a>
        <a href="https://pypi.org/project/llfn/">
          <img src="https://img.shields.io/pypi/dm/llfn?style=for-the-badge" />
        </a>
        <a href="https://github.com/orgexyz/LLFn/network/members">
          <img src="https://img.shields.io/github/forks/orgexyz/LLFn.svg?style=for-the-badge" />
        </a>
        <a href="https://github.com/orgexyz/LLFn/stargazers">
          <img src="https://img.shields.io/github/stars/orgexyz/LLFn.svg?style=for-the-badge" />
        </a>
        <a href="https://github.com/orgexyz/LLFn/issues">
          <img src="https://img.shields.io/github/issues/orgexyz/LLFn.svg?style=for-the-badge" />
        </a>
        <a href="https://github.com/orgexyz/LLFn/blob/master/LICENSE">
          <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" />
        </a>
        <a href="https://github.com/orgexyz/LLFn/blob/master/CODE_OF_CONDUCT.md">
          <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg?style=for-the-badge" />
        </a>
    </div>
    <br />
    <a href="#core-concept-function-is-all-you-need">Core Concept</a>
    ·
    <a href="#installation">Installation</a>
    ·
    <a href="#examples">Examples</a>
    ·
    <a href="https://github.com/orgexyz/LLFn/issues">Report Bug</a>
    ·
    <a href="https://github.com/orgexyz/LLFn/issues">Request Feature</a>
    <br />
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template"><strong>📚 Explore the docs »</strong></a> -->
  </p>
</div>

## Features

**[LLFn](https://github.com/orgexyz/LLFn) has only one goal**: making it 100x easier to experiment and ship AI applications using the LLMs with minimum boilerplate. Even if you're new to LLMs, you can still get started and be productive in minutes, instead of spending hours learning how to use the framework.

- **🔋 Functionify**: turning any `prompt` into a callable function.
- **📤 Flexible Output**: defining LLM result with Python and [`pydantic`](https://github.com/pydantic/pydantic) types.
- **🧱 Infinitely Composable**: any function can call any other function in any order.
- **🛒 Use Any LLM**: leveraging [`LangChain`](https://github.com/hwchase17/langchain)'s LLM interface, drop in your favorite LLM with 100% compatibility.
- **🪶 Light-Weight**: small core framework of LLFn making it extremely easy to understand and extend.

We draw a lot of inspiration from both [FastAPI](https://github.com/tiangolo/fastapi) and [LangChain](https://github.com/hwchase17/langchain), so if you're familiar with either of them you'd feel right at home with LLFn.

## Core Concept: _Function is All You Need_

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

```python
translate("Hello welcome. How are you?", "Thai")
# สวัสดี ยินดีต้อนรับ สบายดีไหม?
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

## Installation

LLFn is available on [PyPi](https://pypi.org/project/llfn/). You can install LLFn using your favorite Python package management software.

```sh
$ pip install llfn # If you use pip
$ poetry add llfn # If you use poetry
```

## Advanced Features

While LLFn is simple and light-weight by design, it does pack some punches above its weight that helps you write AI applications more intuitively.

### 📐 Structured Function Output

You can create a function to automatically format the response into a desired `pydantic` object. To do this, you simply declare the output of your function in `pydantic`, and let the LLFn do the heavy lifting for you.

```python
from pydantic import BaseModel, Field

class AgentLogicOutput(BaseModel):
  tasks: List[str] = Field(..., description="list of tasks to accomplish the given goal")
  reasoning: str  = Field(..., description="why you choose to do these tasks")

@function_prompt
def agent_think(goal: str) -> AgentLogicOutput:
    return f"You're an agent planning 5 tasks to accomplish the following goal: {goal}"

agent_think("Creating online business") # this returns AgentLogicOutput object
```

This allows for powerful composability like tools, agents, etc. without sacrificing the simplicity and verbosity of our program. Under the hood LLFn injects the output format into your prompt and automatically parse the LLM result into the specified format.

### 🗝️ Binding and Overriding LLMs

LLFn let you use both LangChain's [LLMs](https://python.langchain.com/docs/modules/model_io/models/llms/) and [Chat Models](https://python.langchain.com/docs/modules/model_io/models/chat/) to power your functions. There are 2 ways to _assign_ it.

#### Method 1: Binding LLM to All Your Functions

This is the most convenient way to just get your function to work.

```python
from llfn import LLFn
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.7, openai_api_key=API_KEY)

function_prompt = LLFn()
function_prompt.bind(llm)

@function_prompt
def your_function(...):
    ...
```

#### Method 2: Binding/Overriding LLM to A Specific Function

If you want to use different LLMs for different tasks, you can do so by binding `llm` to your function individually.

```python
from langchain.llms import LlamaCpp

llm2 = LlamaCpp(...)

your_function.bind(llm2) # This will override the LLM for this function
```

### 🏄‍♂️ Callbacks and Streaming

#### Callbacks

With LLFn, it's trivial to define callbacks in between execution. Because everything is a function, you can `print` or call any 3rd party APIs anywhere you want.

For example, if you want to print results of executions in between a series of complex AI agent it would look like this.

```python
...

# Assuming that other functions are already defined with function_prompt
@function_prompt
def agent_plan_and_execute(goal: str) -> str:
    tasks = agent_think(goal)
    print(tasks) # Debug
    results = agent_execute_tasks(tasks)
    print(results) # Debug
    evaluation = agent_evaluate_perf(goal, results)
    print(evaluation) # Debug
```

Your function can also take a callback function as a parameter. LLFn does not dictate how you should do it. Anything that works for you would do!

#### Streaming LLM Output

Because LLFn leverages LangChain's LLMs, you can use [LangChain Streaming](https://python.langchain.com/docs/modules/model_io/models/chat/how_to/streaming) directly.

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    ...
)

function_prompt.bind(llm)
```

## Examples

### ⭐️ Single-Prompt Examples

| **Task** | **Showcase** |
|----------|--------------|
| [Machine Translation](/examples/ex_translate_chat_model.py) | Using LangChain Chat Model |
| [Machine Translation](/examples/ex_translate_llm.py)  | Using LangChain LLM Model |
| [Agent Task Planner](/examples/ex_agent_tasks.py) | Defining Structured Output |

### 🌈 Multi-Prompt Complex Examples

| **Task** | **Showcase** |
|----------|--------------|
| [BabyAGI](/examples/ex_babyagi.ipynb) | Implementing BabyAGI with LLFn |

## Sponsors and Contributors

We currently don't take any monetary donations! However, every issue filed and PR are extremely important to us. Here is the roster of contributors and supporters of the project.

<a href="https://orge.xyz"><img height="100" alt="orge.xyz" src="https://github.com/orgexyz/LLFn/assets/891585/3502c0a6-1ab2-445d-9010-1f59f3e23cc0"></a>

<br />

<a href="https://github.com/smiled0g"><img src="https://avatars.githubusercontent.com/smiled0g?v=4" width="50px" alt="smiled0g" /></a>&nbsp;&nbsp;<a href="https://github.com/sorawit"><img src="https://avatars.githubusercontent.com/sorawit?v=4" width="50px" alt="sorawit" /></a>&nbsp;&nbsp;


## Contribution and Feedback

Contributions, feedback, and suggestions are highly encouraged and appreciated! You can contribute to LLFn in the following ways:

- 🔀 Fork the repository and make modifications in your own branch. Then, submit a pull request ([PR](https://github.com/orgexyz/LLFn/pulls)) with your changes.
- ⬇️ Submit issues ([GitHub Issues](https://github.com/orgexyz/LLFn/issues)) to report bugs, suggest new features, or ask questions.

## Citation

If you wish to cite LLFn in your research, we encourage the use of [CITATION.cff](CITATION.cff) provided for appropriate citation formatting. For more details on the citation file format, please visit the [Citation File Format website](https://citation-file-format.github.io).

## License

LLFn is an open-source software under the **MIT license**. This license allows use, modification, and distribution of the software. For complete details, please see the [LICENSE](LICENSE) file in this repository.

## Acknowledgements

We would like to express our gratitude to the following projects and communities for their inspiring work and valuable contributions:

- [LangChain](https://github.com/hwchase17/langchain)
- [Pydantic](https://github.com/pydantic/pydantic)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [OpenAI](https://openai.com/)
