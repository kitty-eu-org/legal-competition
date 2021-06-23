"""
@Time ： 2021/6/23 下午8:56
@Auth ： hzz
@File ：model.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
from datasets import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW,AutoConfig
import torch

from sklearn.metrics import accuracy_score

from my_data_loader import MyDataset
from data_process import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-electra-180g-base-discriminator")

config = AutoConfig.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
config.num_labels = 5
model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-electra-180g-base-discriminator", config=config)
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=2e-5)
train_data_list = handle_data()
res_train_list, res_vail_list = handle_classier_data(train_data_list)
train_dataset = MyDataset(data_list=res_train_list, tokenizer=tokenizer)
vail_dataset = MyDataset(res_vail_list, tokenizer, False)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
vail_loader = DataLoader(vail_dataset, batch_size=4)
total_loss, total_val_loss = 0, 0
total_eval_accuracy = 0

for epoch in range(5):
    for step, batch in tqdm(enumerate(train_loader), desc="training: "):
        model.train()
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        loss.backward()
        optimizer.step()
        if step % 200 == 0 and step > 0:  # 每10步输出一下训练的结果，flat_accuracy()会对logits进行softmax
            model.eval()
            with torch.no_grad():
                logits = torch.nn.Softmax(dim=-1)(logits.data).cpu().numpy()
                # predicted = torch.max(logits, 1)[0]
                predicted = logits[:, 1]
                # logits = logits.detach().cpu().numpy()
                # predicted = predicted.cpu().numpy()
                label_ids = batch["labels"].cpu().numpy()
                accuracy = accuracy_score(labels, predicted)
                print("loss: ", loss)
                print(f"epoch: {epoch}\tstep:{step}\tauc_score: {accuracy:.4f}")
                # 每个epoch结束，就使用validation数据集评估一次模型
    # model.eval()
