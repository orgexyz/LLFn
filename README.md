# LLFn

TBD

## How to use

```py
from llfn import prompt_function


@prompt_function
def translate(input: str, output_language: str) -> str:
    return """
You must automatically detect the language of `input` and tranlate it to `output_language`
"""


@prompt_function
def summarize(input: str, length: int) -> str:
    return """
You must summarize `input` data into a smaller text approximately `length` words long
"""


if __name__ == "__main__":
    print(translate("สวัสดีตอนเช้าครับ อยากรับประทานอะไรดีครับเช้าวันนี้", "english"))
    print(summarize("I love my dogs. They are corgis. They love nuggets", 4))
```

```sh
$ poetry run python example.py
# Good morning, what would you like to eat for breakfast today?
# I love my dogs.
```