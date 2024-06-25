# ========================================================
#             Media and Cognition
#             Homework 3 Support Vector Machine
#             datasets.py - Define the data loader for the traffic sign classification dataset
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2024
# ========================================================


import torch
import torch.utils.data as data


class Traffic_Dataset(data.Dataset):
    def __init__(self, data_root):
        dataset = torch.load(data_root)
        self.datas = dataset["data"]
        self.labels = dataset["label"]

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)
