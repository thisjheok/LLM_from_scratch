# 슬라이딩 윈도우 구현 윈도우 사이즈 만큼 토큰 처리

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# 데이터 셋 생성 클래스 
# [input_batch, target_batch] 2개를 튜플 형태로 반환 
# DataLoader로 배치화하면 각각이 텐서로 쌓임
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "토큰화된 입력의 개수는 적어도 max_length+1과 같아야 합니다."
        
        # 슬라이딩 윈도우 적용 
        # target_chunk는 input_chunk에서 1칸 shift한 형태
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                          stride=128, shuffle=True, drop_last=True,
                          num_workers=0):
    # 토크나이저 초기화
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터 셋 생성
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 데이터 로더 생성
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# raw text 데이터 읽어오기 
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 데이터 로더로 텐서 형태로 로드
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)