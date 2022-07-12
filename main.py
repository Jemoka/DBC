# random for testing
import random

# os for dir
import os

# import pandas
import pandas as pd # type: ignore

# and huggingface
from transformers import BertModel, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding # type: ignore

# torch
import torch
from torch.optim import AdamW # and adam
import torch.nn.functional as F # and functional

# weights and biases
import wandb # type: ignore

# tqdm
from tqdm import tqdm
from wandb.sdk import wandb_run

# import our utils
from util import eval_model_on_batch

# import our training code
from train import train

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# initialize the model
CONFIG = {
    "model": "nghuyong/ernie-2.0-en",
    "batch_size": 64,
    "epochs": 4,
    "lr": 1e-4,
    "max_length": 60,
    "features": ['mor_Utts', 'mor_Words', 'mor_syllables', '#_Prolongation', '%_Prolongation', '#_Broken_word', '%_Broken_word', '#_Block', '%_Block', '#_PWR', '%_PWR', '#_PWR-RU', '%_PWR-RU', '#_WWR', '%_WWR', '#_mono-WWR', '%_mono-WWR', '#_WWR-RU', '%_WWR-RU', '#_mono-WWR-RU', '%_mono-WWR-RU', 'Mean_RU', '#_Phonological_fragment', '%_Phonological_fragment', '#_Phrase_repetitions', '%_Phrase_repetitions', '#_Word_revisions', '%_Word_revisions', '#_Phrase_revisions', '%_Phrase_revisions', '#_Pauses', '%_Pauses', '#_Filled_pauses', '%_Filled_pauses', '#_TD', '%_TD', '#_SLD', '%_SLD', '#_Total_(SLD+TD)', '%_Total_(SLD+TD)', 'Weighted_SLD']
}

DATASET = "./data/transcripts_pauses/alignedpitt-7-11-flucalc-windowed.bat"

# set up the run
# run = wandb.init(project="DBC", entity="jemoka", config=CONFIG)
run = wandb.init(project="DBC", entity="jemoka", config=CONFIG, mode="disabled")

# get the configuration
config = run.config

# Load the current dataset, which is pitt-7-4
df = pd.read_pickle(DATASET)

# combine
df = df[config.features+["split", "utterance", "target"]]

# Split train and test
train_data = df[df["split"] == "train"]
testing_data = df[df["split"] == "test"]

# drop the split column
train_data = train_data.drop(columns=["split"])
testing_data = testing_data.drop(columns=["split"])

# no need for the index anymore
train_data = train_data.reset_index(drop=True)
testing_data = testing_data.reset_index(drop=True)

# create batches
train_batches = train_data.groupby(by=lambda x: int(x % (len(train_data)/config.batch_size)))

test_batches = testing_data.groupby(by=lambda x: int(x % (len(testing_data)/config.batch_size)))

# train our model!
model, tokenizer = train(config.model, train_batches, test_batches, config, wandb_run=run)

# save the model
save_path = f"./models/{run.name}"

# create save path
if not os.path.exists(save_path):
    os.mkdir(save_path)

# save model
torch.save(model, os.path.join(save_path, "model.bin"))
tokenizer.save_pretrained(f"./models/{run.name}")

