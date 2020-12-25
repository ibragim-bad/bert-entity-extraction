import transformers

MAX_LEN = 32
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 6
EPOCHS = 2
BASE_MODEL_PATH = "DeepPavlov/rubert-base-cased-conversational"
MODEL_PATH = "model.bin"
TRAINING_FILE = "data/train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
