import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def process_data(data_path):
    df = pd.read_csv(data_path).dropna()
    # df.loc[:, "sentence_id"] = df["sentence_id"].fillna(method="ffill")

    enc_pos = preprocessing.LabelEncoder()

    df.loc[:, "labels"] = enc_pos.fit_transform(df["labels"])

    sentences = [b for b in batch(df['words'].values), config.MAX_LEN]
    pos = [b for b in batch(df['labels'].values, config.MAX_LEN)]
    return sentences, pos, enc_pos


if __name__ == "__main__":
    sentences, pos, enc_pos = process_data(config.TRAINING_FILE)
    
    meta_data = {
        "enc_pos": enc_pos
    }

    joblib.dump(meta_data, "meta.bin")

    num_pos = len(list(enc_pos.classes_))

    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
    ) = model_selection.train_test_split(sentences, pos, random_state=42, test_size=0.1)

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, pos=train_pos
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sentences, pos=test_pos
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda")
    model = EntityModel(num_pos=num_pos)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        trd = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        trs = engine.eval_fn(valid_data_loader, model, device)
        train_loss, test_loss = trd['loss'], trs['loss']
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        print(trd['cr'])
        print(trs['cr'])
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
