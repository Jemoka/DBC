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

# import our tools
from util import eval_model_on_batch
from train import train

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# initialize the model

# TODO before running:
# ARE WE SURE THESE VALUES ARE THE PARAMETRES WE ARE TESTING??
CONFIG = {
    "model": "nghuyong/ernie-2.0-en",
    "batch_size": 64,
    "epochs": 8,
    "lr": 1e-4,
    "max_length": 60,
    "features": ["%_WWR", "%_mono-WWR", "%_Total_(SLD+TD)"]
}

# that used on helpful-leaf-7
DATASET = "./data/transcripts_pauses/alignedpitt-7-8-flucalc-windowed.bat"

# pg 4, Yuan 2021
K = 50

# set up the run
# run = wandb.init(project="DBC", entity="jemoka", config=CONFIG)
run = wandb.init(project="DBC", entity="jemoka", config=CONFIG, mode="disabled")

# get the configuration
config = run.config

# print metadata to be written down/screenshotted
print(f"\nEvaluation run {run.id}.")
print("---------------------------------------")
print(f"config: {config}")
print(f"data: {DATASET}")
print(f"K: {K}\n")
input("Did you write it down? ")
print("")

#############################

# Load the current dataset, which is pitt-7-4
df = pd.read_pickle(DATASET)

# combine
df = df[config.features+["split", "utterance", "target"]]

# Get the training data
train_data = df[df["split"] == "train"]

# drop the split column
train_data = train_data.drop(columns=["split"])

# no need for the index anymore
train_data = train_data.reset_index(drop=True)

# create k-fold batches
train_folds = train_data.groupby(by=lambda x: int(x % K))

# select each k from 0...K, and make a tuple
train_eval_groupings = []

# create groups
for test_batch in range(K):
    # get train data by selecting everything except for test batch
    train_group_data = train_folds.apply(lambda x,l: (x if x.iloc[0].name != l else None),
                                         test_batch)
    # get test data by... getting test data
    test_group_data = train_folds.get_group(test_batch)

    # reset everybody's index
    train_group_data = train_group_data.reset_index(drop=True)
    test_group_data = test_group_data.reset_index(drop=True)

    # and append
    train_eval_groupings.append((train_group_data, test_group_data))

# results
acc = []
prec = []
recc = []

for i, (train_data, test_data) in enumerate(train_eval_groupings):
    # print
    print(f"\nCurrently evaluating with fold {i}.")

    # create train and test batches
    train_batches = train_data.groupby(by=lambda x: int(x % (len(train_data)/config.batch_size)))
    test_batches = test_data.groupby(by=lambda x: int(x % (len(test_data)/config.batch_size)))
    # train the model!
    model, tokenizer = train(config.model, train_batches, test_data, config, False)

    # and evaluate across batches
    acc_batch = []
    prec_batch = []
    recc_batch = []

    # for each batch
    for batch_id in range(len(test_batches)):
        # get batch
        batch = test_batches.get_group(batch_id)
        # run eval
        res = eval_model_on_batch(model, tokenizer, batch, config.max_length)
        # append results
        acc_batch.append(res[0])
        prec_batch.append(res[1])
        recc_batch.append(res[2])
        
    # and append the averaged results to final list
    acc.append(sum(acc_batch)/len(acc_batch))
    prec.append(sum(prec_batch)/len(prec_batch))
    recc.append(sum(recc_batch)/len(recc_batch))
        
# create a final data frame for results
df = pd.DataFrame({"accuracy": acc, "precision": prec, "recall": recc})

# save the dataframe
df.to_csv(f"./results/results-{K}-fold-{run.id}.csv", index=False)

