"""
@Time ： 2021/6/23 下午9:10
@Auth ： hzz
@File ：my_data_loader.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import functional as F

from jsonpath import jsonpath


class MyDataset(Dataset):
    def __init__(self, data_list, tokenizer, label_flag=True):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.text_list = jsonpath(data_list, "$..text")
        self.encodings = self.tokenizer(self.text_list, truncation=True, padding=True, max_length=128,
                                        return_tensors="pt")
        self.label_flag = label_flag

    def __getitem__(self, idx):
        label = self.data_list[idx]["label"]
        if self.label_flag:
            item_dict = dict()
            for key, val in self.encodings.items():
                item_dict[key] = torch.tensor(val[idx])
                # new_label = F.one_hot(torch.tensor(label), num_classes=5)
                item_dict["labels"] = torch.tensor(label)

            return item_dict
        else:
            item_dict = dict()
            for key, val in self.encodings.items():
                item_dict[key] = torch.tensor(val[idx])
            return item_dict

    def __len__(self):
        length = len(self.data_list)
        return length
