# Byte Pair Encoding : 어휘 사전에 없는 단어를 개별 문자로 분할하여 처리
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special='all')
strings = tokenizer.decode(integers)

print(strings)
