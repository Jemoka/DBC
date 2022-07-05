# random for testing
import random

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

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TOKENIZER = "nghuyong/ernie-2.0-en"
MODEL = "./models/vocal-oath-6"
MAX_LENGTH = 60

#############################

# Load the current dataset, which is pitt-7-4
df = pd.read_pickle("./data/transcripts_nodisfluency/pitt-7-4-windowed.dat")

# Get the testing data
testing_data = df[df["split"] == "test"]

# drop the split column
testing_data = testing_data.drop(columns=["split"])

# no need for the index anymore
testing_data = testing_data.reset_index(drop=True)

#############################

# Epic. Let's load our models.
tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
model = BertForSequenceClassification.from_pretrained(MODEL).to(DEVICE)

#############################

# define validation tools
def eval_model_on_batch(model, batch):
    # encode the batch
    batch_encoded = tokenizer(batch["utterance"].to_list(),
                            return_tensors="pt",
                            max_length=MAX_LENGTH,
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

# run validation
acc, prec, recc = eval_model_on_batch(model, testing_data)
print(f"""
 Model: {Path(MODEL).stem}
 --------------------
 Accuracy: {round(acc, 4)*100:.2f}%
 Precision: {round(prec, 4)*100:.2f}%
 Recall: {round(recc, 4)*100:.2f}%
 F1: {round(2*((prec*recc)/(prec+recc)), 4)*100:.2f}%
""")


