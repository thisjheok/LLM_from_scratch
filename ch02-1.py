import re
# 훈련을 위해 토큰화할 파일을 열기
with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

# 정규 표현식을 이용해서 문장을 토크나이징
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 고유 토큰들에 대해서 리스트를 만들어서 알파벳 순으로 정렬. vocabulary를 만듦
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

# 토큰들에 대해서 번호를 매김
vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break

# 간단한 텍스트 토크나이저 구현 
# 텍스트를 토큰으로 분할하고 어휘사전으로 문자열-정수 매핑을 수행해 토큰 ID를 생성하는 encode 메서드와
# 토큰 ID를 텍스트로 변환하기 위해 역방향으로 정수-문자열 매핑을 수행하는 decode 메서드

class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)

# 어휘 사전에 없는 단어 처리하기 
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

# 어휘 사전에 없는 단어 토크나이징 하기 
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 구둣점 문자 앞의 공백을 삭제합니다.
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(tokenizer.decode(tokenizer.encode(text)))