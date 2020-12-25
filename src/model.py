import config
import torch
import transformers
import torch.nn as nn

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class EntityModel(nn.Module):
    def __init__(self, num_pos):
        super(EntityModel, self).__init__()
        self.num_pos = num_pos
        self.bert = transformers.AutoModel.from_pretrained(config.BASE_MODEL_PATH)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.out_pos = nn.Linear(768, self.num_pos)
    
    def forward(self, ids, mask, token_type_ids, target_pos):
        ol = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)[0]

        bo_pos = self.bert_drop_1(ol) 
        pos = self.out_pos(bo_pos)

        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = loss_pos

        return pos, loss
