---
title: Advanced Features
---

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
