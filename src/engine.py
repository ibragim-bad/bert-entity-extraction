import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    preds = []
    tr = []
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        pos, loss = model(**data)
        tr.append(data['target_pos'].cpu().numpy())
        preds.append(pos.cpu().numpy())
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    preds = np.hstack(preds)
    tr = np.hstack(tr)
    cr = classification_report(tr, preds)
    return {'loss':final_loss / len(data_loader), 'cr':cr}

def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    preds = []
    tr = []

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        pos, loss = model(**data)
        tr.append(data['target_pos'].cpu().numpy())
        preds.append(pos.cpu().numpy())
        final_loss += loss.item()
    preds = np.hstack(preds)
    tr = np.hstack(tr)
    cr = classification_report(tr, preds)
    return {'loss':final_loss / len(data_loader), 'cr':cr}
