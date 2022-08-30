import pandas as pd
import numpy as np
from datasets import load_metric

bleu_metric = load_metric("sacrebleu")

bleu_metric.add(prediction="the the the the the the", reference=["the cat is on the mat"])
results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
results["precisions"] = [np.round(p, 2) for p in results["precisions"]]
print(pd.DataFrame.from_dict(results, orient="index", columns=["Value"]))

bleu_metric.add(prediction="the cat is on mat", reference=["the cat is on the mat"])
results = bleu_metric.compute(smooth_method="floor", smooth_value=0)
results["precisions"] = [np.round(p, 2) for p in results["precisions"]]
print(pd.DataFrame.from_dict(results, orient="index", columns=["Value"]))