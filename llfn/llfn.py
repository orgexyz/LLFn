import json
from functools import update_wrapper, wraps
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class LLFn:
    def __init__(self):
        self.llm = None

    def bind(self, llm):
        self.llm = llm

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            prompt = f(*args, **kwargs)
            if self.llm is None:
                raise ValueError(
                    "You must call global_bind before calling prompt_function"
                )
            output = self.llm.predict_messages(
                [
                    SystemMessage(
                        content=f"""
- You must execute the user command and convert it to exactly one single output. Let's call it X
- X MUST have Python type of `{f.__annotations__['return']}`
- You then MUST output the JSON of X using Python's `json.dumps`
- Executing `json.loads` on the output of your function MUST give exactly X
- You MUST NOT output anything else. No need to provide output context

- For example, if type is `str` you can output `"Hello"` BUT NOT `{{"result": "Hello"}}`
"""
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            return json.loads(output.content)

        return wrapper
