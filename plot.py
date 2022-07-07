# plotting facilities for results
import pandas as pd

# import seaborn
import seaborn as sns
sns.set_theme() # set theme
sns.set_context("poster")
sns.set(rc={'figure.figsize':(1920,1080)})

# and pathlib
from pathlib import Path

# import our utiltiies
from util import mean_confidence_interval

# load results
RESULTS = "./results/results-50-fold-XgsP4FVS6ScFxCZKFJoVQ5.csv"
CONFIDENCE = 0.95

# put it into a dataframe
df = pd.read_csv(RESULTS)

# calculate confidences
acc, acc_conf =  mean_confidence_interval(df.accuracy, CONFIDENCE)
prec, prec_conf =  mean_confidence_interval(df.precision, CONFIDENCE)
recc, recc_conf =  mean_confidence_interval(df.recall, CONFIDENCE)

# plot
plot = sns.FacetGrid(df.melt(var_name="column"), col="column", margin_titles=True)
plot.map(sns.histplot, "value", kde=True)

# Set labels
plot.set_xlabels("value")
plot.set_ylabels("density")
plot.set_titles(col_template="{col_name}")

# save figure
plot.savefig(f"./results/{Path(RESULTS).stem}.png")

# print results
print(f"""
 Results from K-Fold Evaluation
 ------------------------------
 Eval run: {Path(RESULTS).stem}
 Interval Band: {(CONFIDENCE*100):.1f}%

 Accuracy: {(acc*100):.2f}% ± {(acc_conf*100):.2f}%
 Precision: {(prec*100):.2f}% ± {(prec_conf*100):.2f}%
 Recall: {(recc*100):.2f}% ± {(recc_conf*100):.2f}%

 Figure: "./results/{Path(RESULTS).stem}.png"
""")

