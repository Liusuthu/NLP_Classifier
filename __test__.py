# from zhipuai import ZhipuAI
import numpy as np
import torch

# client = ZhipuAI(api_key="80cba4165a53e1602e6d631cbd0caef9.YBb0NWpvyZq7WzIe")

# def get_vector(input):
#     response = client.embeddings.create(
#         model="embedding-2",
#         input=input,
#     )
#     return response.data[0].embedding

# with open("data/train/1.txt",encoding='utf-8') as f:
#     for i,text in enumerate(f):
#         print(i,":")
# A="12"

# print(str(A[0])+str(A[1]))
# a=torch.tensor(1)
# b=torch.tensor(1)
# if(a==b):
#     print('yes')
# vote=[0,0,0,0]
# vote_item="12"
# vote[int(vote_item[1])]=5
# print(vote,vote.index(max(vote)))

a=torch.tensor([[1,2,3]])
b=torch.tensor([[1,2],[2,3],[3,4]])
print(a.shape,b.shape)
print(a@b)