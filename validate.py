# random for testing
import random

# os for pathing
import os

# import pandas
import pandas as pd # type: ignore

# and huggingface
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding # type: ignore

# torch
import torch
import torch.nn.functional as F # and functional

# tqdm
from tqdm import tqdm

# pathlib
from pathlib import Path

# our utils
from util import eval_model_on_batch

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TOKENIZER = "./models/silver-sky-42"
MODEL = "./models/silver-sky-42"
FEATURES = ["%_WWR", "%_mono-WWR", "%_Total_(SLD+TD)"]
MAX_LENGTH = 60
WINDOW_SIZE = 5

#############################

# Load the current dataset, which is pitt-7-4
df = pd.read_pickle("./data/transcripts_pauses/alignedpitt-7-8-flucalc-windowed.bat")

# combine
df = df[FEATURES+["split", "utterance", "target"]]

# Get the testing data
testing_data = df[df["split"] == "test"]

# drop the split column
testing_data = testing_data.drop(columns=["split"])

# no need for the index anymore
testing_data = testing_data.reset_index(drop=True)

#############################

# Epic. Let's load our models.
tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
model = torch.load(os.path.join(MODEL, "model.bin")).to(DEVICE)

model.eval()

#############################

# run validation
acc, prec, recc = eval_model_on_batch(model, tokenizer, testing_data, MAX_LENGTH)
print(f"""
 Model: {Path(MODEL).stem}
 --------------------
 Accuracy: {round(acc, 4)*100:.2f}%
 Precision: {round(prec, 4)*100:.2f}%
 Recall: {round(recc, 4)*100:.2f}%
 F1: {round(2*((prec*recc)/(prec+recc)), 4)*100:.2f}%
""")

