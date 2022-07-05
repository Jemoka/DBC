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

# tqdm
from tqdm import tqdm

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

    # encode the batch
    batch_encoded = tokenizer(batch["utterance"].to_list(),
                            return_tensors="pt",
                            max_length=max_length,
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

# define prediction on sample tools
def predict_on_sample(model, sample, tokenizer, max_length):
    """evaluate a pytorch model on a batch of data

    Args:
        model (torch.Model): the model to eval
        sample (str): string to test
        batch (pd.DataFrame): input batch
        max_length (int): the max length the model is trained with

    Returns:
        list[float] [control_logit, ad_logit]
    """

    # encode the batch
    batch_encoded = tokenizer([sample],
                            return_tensors="pt",
                            max_length=max_length,
                            padding=True,
                            truncation=True).to(DEVICE)
    # pass it through the model
    model_output = model(**batch_encoded)["logits"].detach()
    # print results
    return model_output[0].cpu().numpy().tolist()

