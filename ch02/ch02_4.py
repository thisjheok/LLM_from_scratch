# 2.7 토큰 임베딩 만들기 
# 임베딩 층을 사용하여 연속적인 벡터 표현으로 임베딩 해야함
# 임베딩 층은 기본적으로 룩업 연산, 사전에 생성한 weight 테이블을 만들고 이를 기반으로 토큰 ID에 따른 가중치를 할당(인덱싱)

import torch
# 예시 샘플, 하나의 샘플에 토큰이 4개 
input_ids = torch.tensor([2, 3, 5, 1])

# 6x3 가중치 행렬 생성
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("[Embedding layer weight]:",embedding_layer.weight)

# 토큰 ID 3을 3차원 벡터로 변환, 가중치 행렬에서 index 3을 찾는다.(룩업 연산)
print("[token ID 3]:",embedding_layer(torch.tensor([3])))
# 샘플 내의 모든 토큰(4개)에 대해서:
# 가중치 행렬에서 샘플 내에 해당하는 것만 출력한다.  
print("[input ids 임베딩]:",embedding_layer(input_ids))

# 2.8 단어 위치 임베딩하기
# 임베딩 층은 토큰 ID를 입력 시퀀스에서 어떤 위치에 있던지 상관없이 동일한 벡터 표현으로 바꾼다.
# 문제점: 입력 시퀀스에서 토큰 ID가 구분되지 않으므로, 문장안에서 역할/순서 고려가 되지 않는다. 
# 따라서, 위치 임베딩 값을 더해주어, 문장 내의 역할/순서를 반영해준다.

# 바이트 페어 인코더 어휘 사전 크기 기반 인코딩 가정: 50257x256 가중치 행렬 생성
vocab_size = 50257
output_dim = 256

# 50257x256 가중치 행렬 생성
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 데이터 로더에서 데이터 샘플링
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

from ch02_3 import create_dataloader_v1
from torch.utils.data import Dataset, DataLoader

# input data는 8x4 행렬 
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# 인풋에 대한 임베딩 가중치 행렬을 생성. 8x4x256 행렬 
token_embeddings = token_embedding_layer(inputs)

# 위치 임베딩 행렬을 만들기 위해 또 다른 임베딩 층을 생성, 4x256 이를 하나의 샘플 내의 각각의 토큰 임베딩 값에 더해줌 
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
print("pos_embedding_layer",pos_embedding_layer.weight)

# 위치 번호에 따라서 임베딩 시킴; 0,1,2,...,max_length-1 위치 인덱스 => 각 위치 번호에 해당하는 벡터로 변환
pos_embeddings = pos_embedding_layer(torch.tensor([3, 2, 1, 0])) # torch.arange(max_length) 하면 0 1 2 3으로 위치 번호 텐서 적용
# torch.tensor([3, 2, 1, 0])로 주면 역순으로 각 텐서 요소에 해당하여 알맞게 배치되어있는 것을 확인 가능 
print("pos_embeddings",pos_embeddings)

# 트랜스포머에 들어갈 최종 입력 임베딩: 토큰 임베딩 + 위치 임베딩
input_embeddings = token_embeddings + pos_embeddings