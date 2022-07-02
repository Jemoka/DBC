# os utilities
import os

# random for shuffle!
import random

# import pathlib
from pathlib import Path

# import globbing tools
import glob

# import pandas
import pandas as pd # type: ignore

# import transformers
from transformers import AutoModel, AutoTokenizer # type: ignore

#################################################

# set the path for data, this changes based on
# the experiement we are running
DATA_PATH =  "./data/transcripts_nodisfluency/pitt-7-1/"

# glob for the control files and dementia files
control_flo_outs = glob.glob(os.path.join(DATA_PATH, "control/*.txt"))
dementia_flo_outs = glob.glob(os.path.join(DATA_PATH, "dementia/*.txt"))

#################################################

# read the files as a csv and clean

def read_and_clean(files, target):
    """read a list of .txt utterances and clean them into a dataframe

    Arguments:
        files (list[string]): list of files to clean
        target (any): the thing to go into the target column

    Returns:
        a DataFrame with columns trial utterance and target
        specificed in the "target" argument

    """

    # create an array for dfs
    dfs = []

    # for each control file
    for data_file in files:
        # read the datafile as a single column csv
        data = pd.read_csv(data_file, sep="|", header=None)
        # label column as utterance
        data.columns = ["utterance"]
        # add another column representing file name
        data["trial"]=Path(data_file).stem
        # and another representing target
        data["target"] = target
        # let's reorder the columns a little
        data = data[["trial", "utterance", "target"]]
        # append!
        dfs.append(data)

    # finally, assemble control data
    result = pd.concat(dfs)

    # lastly, clean the utterance by strip
    result["utterance"] = result["utterance"].apply(lambda x:x.strip())

    return result

# note that, for the "target" colmumn, 1 is Dementia and 0 is control

## control files ##
control = read_and_clean(control_flo_outs, 0)

## dementia files ##
dementia = read_and_clean(dementia_flo_outs, 1)

# concat!
data = pd.concat([control, dementia])
# reset index
data = data.reset_index(drop=True)

# finally, make `trial` a part of the multiindex
# set the index to the multiindex
data_indicies = zip(data["trial"], data.index)
data.index = pd.MultiIndex.from_tuples(data_indicies, names=["trial", "sample"])
# drop the existing trial column
data = data.drop(columns=["trial"])

# inspecting the data
num_controls = len(data[data["target"] == 0]) # 1921 controls
num_dementia = len(data[data["target"] == 1]) # 3736 dementia
desired_length = min(num_controls, num_dementia)

# get indicies
control_indicies = set([i[0] for i in data[data["target"] == 0].index])
dementia_indicies = set([i[0] for i in data[data["target"] == 1].index])

# shuffle
random.shuffle(list(control_indicies))
random.shuffle(list(dementia_indicies))

# and shuffle actually
data = data.loc[list(control_indicies)+list(dementia_indicies)]

# finally, crop 
data_controls = data[data["target"] == 0].iloc[0:desired_length]
data_dementia = data[data["target"] == 1].iloc[0:desired_length]

# concat back
data = pd.concat([data_controls, data_dementia])
data

