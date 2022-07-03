# import pandas
import pandas as pd # type: ignore

# and huggingface
from transformers import AutoTokenizer, AutoModel # type: ignore

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




