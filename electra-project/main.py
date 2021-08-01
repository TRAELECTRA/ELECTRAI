import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import ElectraModel, ElectraTokenizer

model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")



class ElectraForNER(nn.Module):
    def __init__(self, config):
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(in_features=config.hidn_dim, out_features=config.label_num)

        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(self, input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               labels=None):

        outputs = self.electra(input_ids)
        outputs = self.dropout(outputs)
        logits = self.linear(outputs)

        if attention_mask is None:
            loss = self.loss_fct(np.argmax(logits), labels)
        else:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = self.loss_fct(active_logits, active_labels)

        return logits, loss


'''
1) add layer for finetuning 
2) data processing 
3) finetuning 실제 코드 2를 거쳐서 1을 시행할 수 있도록 하는 코드 작성 
'''


def main_finetune():
    # argparse configs/config.json

    # data file load and data processing (dataloader 등)

    # model 불러오기

    # 학습
    '''

    for e in epoch:
        for i, batch in enumerate(dataloader): >> 데이터를 먼저 만들고 정해야됨
            # train

            # 특정 step에서 valid(evaluate)
             성능이 좋은 top-k개의 모델 저장

    '''


if __name__ == '__main__':
    main_finetune()
