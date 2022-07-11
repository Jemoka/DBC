# random for testing
import random

# import pandas
import pandas as pd # type: ignore

# import numpy
import numpy as np

# and huggingface
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding # type: ignore

# torch
import torch
from torch.optim import AdamW # and adam
import torch.nn.functional as F # and functional

# tqdm
from tqdm import tqdm

# stats
import scipy.stats

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# define validation tools
def eval_model_on_batch(model, tokenizer, batch, max_length):
    """evaluate a pytorch model on a batch of data

    Args:
        model (torch.Model): the model to eval
        tokenizer (transformers.Tokenizer): tokenizer
        batch (pd.DataFrame): input batch
        max_length (int): the max length the model is trained with

    Returns:
        (float)s accuracy, precision, recall
    """

    # get meta features
    batch_meta_features = torch.tensor(batch.drop(columns=["utterance",
                                                        "target"]).to_numpy())
    batch_meta_features = batch_meta_features.to(DEVICE).float()

    # encode the batch
    batch_encoded = tokenizer(batch["utterance"].to_list(),
                            return_tensors="pt",
                            max_length=max_length,
                            padding=True,
                            truncation=True).to(DEVICE)

    # pass it through the model
    model_output = model(**batch_encoded,
                         meta_features=batch_meta_features)["logits"].detach()
    model_output_encoded = (model_output > 0.5).squeeze()

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

# confidence intervals
def mean_confidence_interval(data, confidence=0.95):
    """calculate an n confidence interval

    Arguments:
        data (array-like): array-like data
        [confidence] (float): interval

    Returns:
    (float)s mean and confidence band

    Source: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


