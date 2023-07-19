import dotenv
import os
from functools import wraps
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


dotenv.load_dotenv()

llm = ChatOpenAI(
    temperature=0.8,
    model=os.getenv("OPENAI_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)  # type: ignore


def prompt_function(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        prompt = f(*args, **kwargs)
        output = llm.predict_messages(
            [
                SystemMessage(
                    content=f"You are a function intepreter. The function annotation is: {f.__annotations__}"
                ),
                SystemMessage(
                    content=f"You MUST output exactly the `json.dumps` of the `return` and nothing else. MUST NOT have any other context"
                ),
                SystemMessage(content=prompt),
                HumanMessage(content=f"args: {args}, kwargs: {kwargs}"),
            ]
        )
        import json

        return json.loads(output.content)

    return wrapper
