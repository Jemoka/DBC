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

# initialize the model
CONFIG = {
    "model": "nghuyong/ernie-2.0-en",
    "batch_size": 8,
    "epochs": 2,
    "lr": 3e-3,
    "max_length": 60
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

# and also our optimizer
optim = AdamW(model.parameters(), lr = config.lr)

# watch!
run.watch(model)

#############################

# ok, time for training

model.train()

# for each epoch
for epoch in range(config.epochs):

    # print current training
    print(f"training epoch {epoch}")

    for batch_id in tqdm(range(train_batch_count)):
        # get the batch
        batch = train_batches.get_group(batch_id)

        # encode the batch
        batch_encoded = tokenizer(batch["utterance"].to_list(),
                                return_tensors="pt",
                                max_length=config.max_length,
                                padding=True,
                                truncation=True)

        # encode the labels
        labels_encoded = F.one_hot(torch.tensor(batch["target"].to_numpy()), num_classes=2)

        # run the model
        model_output = model(**batch_encoded, labels=labels_encoded.float())

        # backprop the loss
        model_output["loss"].backward()

        # and update the model
        optim.step()
        optim.zero_grad()

        # plotting to training graph
        run.log({
            "loss": model_output["loss"].item()
        })

# save the model
model.save_pretrained(f"./models/{run.name}")

