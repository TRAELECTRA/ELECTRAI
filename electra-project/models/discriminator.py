import numpy as np
import torch
import torch.nn as nn
from models.common import Transformer, LayerNorm

class Discriminator(nn.Module):
    def __init__(self, Config):
        super().__init__()
        self.transformer = Transformer(Config)  # 기본 코드에서 트랜스포머를 가져온다. -> 이제 여기에 Generator Head라고 되어있는 애의 내용물을 붙여주면 된다.
        self.dense = nn.Linear(Config.dim,
                               Config.dim)  # 가중치를 N개로 고려를 다 해서 전처리를 해준다? 이전에는 하나의 노드가 K개만 소통하고 있기 때문에, dense를 하기 전과 후의 가중치가 달라지므로.
        # Linear 자체가 Weight를 다시 계산을 해주는 거니까.
        # self.activation = F.gelu
        # self.norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.classifier = nn.Linear(Config.dim, 1)  # Original Replace / 2차원으로 하면 되지 않나?

    def forward(self, x, seg, is_replaced_label=None, input_mask=None):
        hidden_states = self.transformer(x, seg, input_mask)  # Batch * Sequence Length * Dim

        h = self.dense(hidden_states)  # 처음에 댄스에 주어진 히든 스테이트가 주어진다. 이 히든 스테이트는 아마 self.transformer를 거쳐서 나온 애일 것. 히든 스테이트
        # -> Batch * Sequence Length * Config.Dim

        h = self.activation(h)

        h = self.norm(h)
        logits = self.classifier(hidden_states)
        # Classifier ->Batch * Sequence Length * 1  == logits
        outputs = (logits,)

        if is_replaced_label is not None:
            loss_fct = nn.BCEWithLogitsLoss()

            if input_mask is not None:
                # active_loss.shape = (-1, Sequence Length) -> 1인 애들만 찾아서 True False 행렬로 바꿔준다.
                active_loss = input_mask.view(-1, hidden_states.shape[1]) == 1

                # active_logits = Batch * Sequence Length * 1 -> (-1, Sequence Length) * [true false와 곱해준다.]
                active_logits = logits.view(-1, hidden_states.shape[1])[active_loss]  # false인 친구들은 이제 사라짐 ㅠ
                active_labels = is_replaced_label[
                    active_loss]  # 그리고 이제 여기서 True False인 애들을 남겨서 -> 주어진 정답인 is_replaced_label과 곱해주면 active_labesl가 나오고
                disc_loss = loss_fct(active_logits,
                                     active_labels.float())  # 그리고 이제 여기서 loss_function을 먹여서 disc_loss를 계산해주고
            else:
                disc_loss = loss_fct(logits.view(-1, hidden_states.shape[1]), is_replaced_label.float())

            outputs += (disc_loss,)  # 아웃풋에 더해주고

        return outputs  # 보내줍니다.