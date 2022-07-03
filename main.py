# random for testing
import random

# import pandas
import pandas as pd # type: ignore

# and huggingface
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding # type: ignore

# weights and biases
import wandb # type: ignore

# initialize the model
CONFIG = {
    "model": "nghuyong/ernie-2.0-en",
    "batch_size": 8
}

# set up the run
# run = wandb.init(project="DBC", entity="jemoka", config=CONFIG)
run = wandb.init(project="DBC", entity="jemoka", config=CONFIG, mode="disabled")

# get the configuration
config = run.config

#############################

# Load the current dataset, which is pitt-7-1
df = pd.read_pickle("./data/transcripts_nodisfluency/pitt-7-1-windowed.dat")

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
train_batch_count = len(train_batches) - 1 # minus one to drop half-batch

test_batches = train_data.groupby(by=lambda x: int(x % (len(testing_data)/config.batch_size)))
test_batch_count = len(test_batches) - 1  # minus one to drop half-batch

#############################

# Epic. Let's load our models.
tokenizer = BertTokenizer.from_pretrained(config.model)
model = BertForSequenceClassification.from_pretrained(config.model)

# train_data

