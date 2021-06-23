"""
@Time ： 2021/6/23 下午3:15
@Auth ： hzz
@File ：data_process.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import json
import os

project_dir = os.path.dirname(__file__)


def handle_data(train=True):
    if train:
        data_base_path = os.path.join(project_dir, "data/train")
    else:
        data_base_path = os.path.join(project_dir, "data/test")
    file_name_list = os.listdir(data_base_path)
    file_abs_name_list = [os.path.join(data_base_path, i) for i in file_name_list]
    res_list = []
    for i in file_abs_name_list:
        f = open(i, mode='r', encoding="utf-8")
        res_list.append(json.load(f))
    return res_list


def handle_classier_data(res_list, vail_size=0.2):
    """
    为分类任务处理数据
    :param res_list: 前面处理好的数据
    :return:
    """
    length = len(res_list)
    train_length = int(length * (1 - vail_size))
    train_list = res_list[:train_length]
    vail_list = res_list[train_length:]
    res_train_list = []
    res_vail_list = []
    for train_pair in train_list:
        answer = train_pair["answer"]
        question = train_pair["question"]
        cause = train_pair["cause"]
        text = answer + "," + question
        label = cause
        res_train_list.append({"text": text,"label": label})
    for vail_pair in vail_list:
        answer = vail_pair["answer"]
        question = vail_pair["question"]
        cause = vail_pair["cause"]
        text = answer + "," + question
        label = cause
        res_vail_list.append({"text": text, "label": label})
    return res_train_list, res_vail_list
