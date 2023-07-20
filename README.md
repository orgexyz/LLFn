# LLFn

TBD

## How to use

````py
import dotenv
import os
from llfn import LLFn
from langchain.chat_models import ChatOpenAI


prompt_function = LLFn()


@prompt_function
def translate(text: str, to_language: str) -> str:
    return f"""
You must automatically detect the language of the following text and tranlate it to {to_language}
```
{text}
```
"""


@prompt_function
def summarize(text: str, length: int) -> str:
    return f"""
You must summarize the following text to a smaller text approximately {length} words long
```
{text}
```
"""

if __name__ == "__main__":
    dotenv.load_dotenv()
    llm = ChatOpenAI(
        temperature=0.8,
        model=os.getenv("OPENAI_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )  # type: ignore
    prompt_function.bind(llm)
    print(translate("สวัสดีตอนเช้าครับ อยากรับประทานอะไรดีครับเช้าวันนี้", "english"))
    print(summarize("I love my dogs. They are corgis. They love nuggets", 4))
````

```sh
$ poetry run python example.py
# Good morning, what would you like to have for breakfast today?
# I love corgi dogs
```