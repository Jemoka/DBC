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

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# initialize the model
CONFIG = {
    "model": "nghuyong/ernie-2.0-en",
    "batch_size": 4,
    "epochs": 3,
    "lr": 1e-5,
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

test_batches = testing_data.groupby(by=lambda x: int(x % (len(testing_data)/config.batch_size)))
test_batch_count = len(test_batches) - 1  # minus one to drop half-batch

#############################

# Epic. Let's load our models.
tokenizer = BertTokenizer.from_pretrained(config.model)
model = BertForSequenceClassification.from_pretrained(config.model).to(DEVICE)

# and also our optimizer
optim = AdamW(model.parameters(), lr = config.lr)

# watch!
run.watch(model)

#############################

# define validation tools
def eval_model_on_batch(model, batch):
    # encode the batch
    batch_encoded = tokenizer(batch["utterance"].to_list(),
                            return_tensors="pt",
                            max_length=config.max_length,
                            padding=True,
                            truncation=True).to(DEVICE)

    # pass it through the model
    model_output = model(**batch_encoded)["logits"].detach()
    model_output_encoded = model_output.argmax(dim=1)

    # get targets
    targets = torch.Tensor(batch["target"].to_numpy()).to(DEVICE)

    # calculate accuracy, precision, recall
    # calculate pos/neg/etc.
    true_pos = torch.logical_and(model_output_encoded == targets,
                                model_output_encoded.bool())
    true_neg = torch.logical_and(model_output_encoded == targets,
                                torch.logical_not(model_output_encoded.bool()))
    false_pos = torch.logical_and(model_output_encoded != targets,
                                model_output_encoded.bool())
    false_neg = torch.logical_and(model_output_encoded != targets,
                                torch.logical_not(model_output_encoded.bool()))

    # create the counts
    true_pos = torch.sum(true_pos).cpu().item()
    true_neg = torch.sum(true_neg).cpu().item()
    false_pos = torch.sum(false_pos).cpu().item()
    false_neg = torch.sum(false_neg).cpu().item()

    acc = (true_pos+true_neg)/len(targets)
    if (true_pos+false_pos) == 0:
        prec = 0
    else:
        prec = true_pos/(true_pos+false_pos)

    if (true_pos+false_neg) == 0:
        recc = 0
    else:
        recc = true_pos/(true_pos+false_neg)

    # and return
    return acc, prec, recc

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
                                truncation=True).to(DEVICE)

        # encode the labels
        target_tensor = torch.tensor(batch["target"].to_numpy()).to(DEVICE)
        labels_encoded = F.one_hot(target_tensor, num_classes=2)

        # run the model
        model_output = model(**batch_encoded, labels=labels_encoded.float())

        # backprop the loss
        model_output["loss"].backward()

        # calculate the accuracy
        model_output_encoded = model_output["logits"].detach().argmax(dim=1)
        acc = torch.sum(model_output_encoded.bool() == target_tensor)/len(target_tensor)

        # and update the model
        optim.step()
        optim.zero_grad()

        # plotting to training graph
        run.log({
            "loss": model_output["loss"].cpu().item(),
            "acc": acc.cpu().item()
        })

        # for every 10 batches, randomly perform a single validation sample
        if batch_id % 10 == 0:
            # run validation 
            val_batch = test_batches.get_group(random.randint(0, len(test_batches)-1))
            acc, prec, recc = eval_model_on_batch(model, val_batch)

            # plot
            run.log({
                "val_accuracy": acc,
                "val_prec": prec,
                "val_recc": recc,
            })
        

# save the model
model.save_pretrained(f"./models/{run.name}")

