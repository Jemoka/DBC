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

# intepretation tools
from transformers_interpret import SequenceClassificationExplainer

# our utils
from util import predict_on_sample, eval_model_on_batch

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TOKENIZER = "./models/royal-pond-21"
MODEL = "./models/royal-pond-21"
MAX_LENGTH = 60
WINDOW_SIZE = 5

#############################

# Load the current dataset, which is pitt-7-4
df = pd.read_pickle("./data/transcripts_pauses/alignedpitt-7-8-windowed.bat")

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
explainer = SequenceClassificationExplainer(model, tokenizer)

#############################


# run predction interactively
while True:
    # take sample
    sample:list = []

    # get input until its filled
    while len(sample) < WINDOW_SIZE: 
        sample.append(input(f"{len(sample)}> ").strip().lower())

    # continue to validate on test set if needed
    if sample[-1] == "v":
        break

    # get results
    result = predict_on_sample(model, " ".join(sample), tokenizer, MAX_LENGTH)

    # get explain attrs 
    attrs = explainer(" ".join(sample))

    # round the attr results
    attrs_rounded = [(i[0], round(i[1],2)) for i in attrs]

    # print results
    print(f"""
Model: {Path(MODEL).stem}
--------------------
Sample: {" ".join(sample)}
Conclusion: {"dementia" if result[1] > result[0] else "control"}
Preds: {result}
--------------------

Token Predictions
    """)

    # print tokens
    for token in attrs_rounded:
        # print it, aligned
        print(f"{token[0]:<12}{token[1]:>5}")

    # print a newline
    print()

#############################

# run validation
acc, prec, recc = eval_model_on_batch(model, tokenizer, testing_data, MAX_LENGTH)
print(f"""
 Model: {Path(MODEL).stem}
 --------------------
 Accuracy: {round(acc, 4)*100:.2f}%
 Precision: {round(prec, 4)*100:.2f}%
 Recall: {round(recc, 4)*100:.2f}%
 F1: {round(2*((prec*recc)/(prec+recc)), 4)*100:.2f}%
""")

