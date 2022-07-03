# os utilities
import os
from posixpath import dirname

# random for shuffle!
import random

# import pathlib
from pathlib import Path

# import globbing tools
import glob

# import pandas
import pandas as pd # type: ignore

#################################################

# set the path for data, this changes based on
# the experiement we are running
DATA_PATH =  "./data/transcripts_nodisfluency/pitt-7-1/" # in path
OUT_PATH = "./data/transcripts_nodisfluency/pitt-7-1.dat" # out path
TESTING_SPLIT = 5  # testing split (patients kper class)

#################################################

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

#################################################

# finally, make `trial` a part of the multiindex
# set the index to the multiindex
data_indicies = zip(data["trial"], data.index)
data.index = pd.MultiIndex.from_tuples(data_indicies, names=["trial", "sample"])
# drop the existing trial column
data = data.drop(columns=["trial"])

#################################################

# get indicies
control_indicies = list(set([i[0] for i in data[data["target"] == 0].index]))
dementia_indicies = list(set([i[0] for i in data[data["target"] == 1].index]))

# shuffle
randomness = random.Random(0)
randomness.shuffle(control_indicies)
randomness.shuffle(dementia_indicies)

# crop out the indicies to ensure length
desired_length = min(len(control_indicies), len(dementia_indicies))
control_indicies = control_indicies[:desired_length]
dementia_indicies = dementia_indicies[:desired_length]

# parcel out the last TESTING_SPLIT from each for testing
training_indicies = control_indicies[:-TESTING_SPLIT]+dementia_indicies[:-TESTING_SPLIT]
testing_indicies = control_indicies[-TESTING_SPLIT:]+dementia_indicies[-TESTING_SPLIT:]

# and shuffle actually
training_data = data.loc[training_indicies]
testing_data = data.loc[testing_indicies]

training_data["split"] = "train"
testing_data["split"] = "test"

# concatenate
data = pd.concat([training_data, testing_data])

# final shuffle
data = data.iloc[randomness.sample(range(len(data)), len(data))]

# dump to data
data.to_pickle(OUT_PATH)

