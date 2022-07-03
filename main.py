# import pandas
import pandas as pd # type: ignore

# and huggingface
from transformers import BertForSequenceClassification, BertTokenizer # type: ignore

# weights and biases
import wandb # type: ignore

# initialize the model
CONFIG = {
    "model": "nghuyong/ernie-2.0-en"
}

# set up the run
# run = wandb.init(project="DBC", entity="jemoka", config=CONFIG)
run = wandb.init(project="DBC", entity="jemoka", config=CONFIG, mode="disabled")

# get the configuration
config = run.config

#############################

# Load the current dataset, which is pitt-7-1
df = pd.read_pickle("./data/transcripts_nodisfluency/pitt-7-1.dat")

# Split train and test
train_data = df[df["split"] == "train"]
testing_data = df[df["split"] == "test"]

# drop the split column
train_data = train_data.drop(columns=["split"])
testing_data = testing_data.drop(columns=["split"])

# Epic. Let's load our models.
tokenizer = BertTokenizer.from_pretrained(config.model)
model = BertForSequenceClassification.from_pretrained(config.model)


train_data

