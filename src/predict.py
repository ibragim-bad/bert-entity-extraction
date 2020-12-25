import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel


if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]

    num_pos = len(list(enc_pos.classes_))

    device = torch.device("cuda")
    model = EntityModel(num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    print('please type')

    sentence = """
    привет как дела а я вот сходил недавно к врачу чтобы узнать как это
    """
    while sentence != 'q':
        print('\n>')
        sentence = input()

        tokenized_sentence = config.TOKENIZER.encode(sentence)

        sentence = sentence.split()
        print(sentence)
        print(tokenized_sentence)

        test_dataset = dataset.EntityDataset(
            texts=[sentence], 
            pos=[[0] * len(sentence)]
        )

        with torch.no_grad():
            data = test_dataset[0]
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            pos, _ = model(**data)
            print(
                enc_pos.inverse_transform(
                    pos.argmax(2).cpu().numpy().reshape(-1)
                )[:len(tokenized_sentence)]
            )
