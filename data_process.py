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
    file_abs_name_list = [(i.split(".")[0], os.path.join(data_base_path, i)) for i in file_name_list]
    res_list = []
    for i in file_abs_name_list:
        id = i[0]
        f = open(i[1], mode='r', encoding="utf-8")
        a_dict = json.load(f)
        a_dict["id"] = id
        res_list.append(a_dict)
    return res_list


def handle_classier_data(res_list, vail_size=0.2):
    """
    为分类任务处理数据
    :param res_list: 前面处理好的数据
    :return:
    """
    label_dict = {'房产纠纷': 0, '劳动纠纷': 1, '交通事故': 2, '债权债务': 3, '婚姻家庭': 4}
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
        label = label_dict[cause]
        res_train_list.append({"text": text, "label": label, "id": train_pair["id"]})
    for vail_pair in vail_list:
        answer = vail_pair["answer"]
        question = vail_pair["question"]
        cause = vail_pair["cause"]
        text = answer + "," + question
        label = label_dict[cause]
        res_vail_list.append({"text": text, "label": label, "id": vail_pair["id"]})
    return res_train_list, res_vail_list


if __name__ == '__main__':
    train_data_list = handle_data()
    res_train_list, res_vail_list = handle_classier_data(train_data_list)
    # import jsonpath
    #
    # print(set(jsonpath.jsonpath(train_data_list, "$..cause"))
    #
    #       )
    import pandas as pd

    train_df = pd.DataFrame(res_train_list)
    train_df["text_length"] = train_df["text"].str.len()
    print(train_df["label"])
