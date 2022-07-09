# random for testing
import random

# import pandas
import pandas as pd # type: ignore

# and huggingface
from transformers import BertForSequenceClassification, BertTokenizer
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
    "epochs": 3,
    "lr": 1e-4,
    "max_length": 60
}

DATASET = "./data/transcripts_pauses/alignedpitt-7-8.bat"

# set up the run
# run = wandb.init(project="DBC", entity="jemoka", config=CONFIG)
run = wandb.init(project="DBC", entity="jemoka", config=CONFIG, mode="disabled")

# get the configuration
config = run.config

# Load the current dataset, which is pitt-7-4
df = pd.read_pickle(DATASET)

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
model, _ = train(config.model, train_batches, test_batches, config, wandb_run=run)

# save the model
model.save_pretrained(f"./models/{run.name}")

