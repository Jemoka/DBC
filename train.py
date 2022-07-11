# random for testing
import random

# import numpy and pandas
import pandas as pd # type: ignore
import numpy as np

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

# import our utils 
from util import eval_model_on_batch

# import our models
from model import Model

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training code called by both k-fold as well as one-shot holdout
def train(base_model, train_batches, test_batches, config, run_val=True, wandb_run=None):

    # set up dummy run if needed
    if wandb_run:
        run = wandb_run
    else:
        run = run = wandb.init(project="DBC", entity="jemoka", mode="disabled")

    # count batches 
    train_batch_count = len(train_batches) # minus one to drop half-batch
    test_batch_count = len(test_batches)  # minus one to drop half-batch

    # Let's load our tokenizer.
    tokenizer = BertTokenizer.from_pretrained(base_model)

    # and let's load our model by calculating feature size
    num_features = len(train_batches.get_group(0).columns)
    model = Model(base_model, num_features-2).to(DEVICE)

    # resize model to add pause token
    tokenizer.add_tokens(["[pause]"])
    model.base_model.resize_token_embeddings(len(tokenizer))

    # and also our optimizer
    optim = AdamW(model.parameters(), lr = config.lr)

    # watch!
    run.watch(model)

    # ok, time for training
    model.train()

    # for each epoch
    for epoch in range(config.epochs):

        # print current training
        print(f"training epoch {epoch}")

        for batch_id in tqdm(range(train_batch_count)):
            # get the batch
            batch = train_batches.get_group(batch_id)

            # get meta features
            batch_meta_features = torch.tensor(batch.drop(columns=["utterance",
                                                                "target"]).to_numpy())
            batch_meta_features = batch_meta_features.to(DEVICE).float()

            # encode the batch
            batch_encoded = tokenizer(batch["utterance"].to_list(),
                                        return_tensors="pt",
                                        max_length=config.max_length,
                                        padding=True,
                                        truncation=True).to(DEVICE)

            # encode the labels
            target_tensor = torch.unsqueeze(torch.tensor(batch["target"].to_numpy()),
                                            1).to(DEVICE)
            labels_encoded = F.one_hot(target_tensor, num_classes=2)

            # run the model
            model_output = model(**batch_encoded,
                                 meta_features=batch_meta_features,
                                 labels=labels_encoded.float())

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
            if batch_id % 10 == 0 and run_val:
                # run validation 
                val_batch = test_batches.get_group(random.randint(0, len(test_batches)-1))
                acc, prec, recc = eval_model_on_batch(model, tokenizer, val_batch, config.max_length)

                # plot
                run.log({
                    "val_accuracy": acc,
                    "val_prec": prec,
                    "val_recc": recc,
                })


    # save the model
    return model, tokenizer

