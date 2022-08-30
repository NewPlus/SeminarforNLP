import pandas as pd
import numpy as np
import nltk
from datasets import load_metric
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import pipeline, set_seed

dataset = load_dataset("cnn_dailymail", version="3.0.0")
sample = dataset["train"][1]
sample_text = dataset["train"][1]["article"][:2000]

# We'll collect the generated summaries of each model in a dictionary
summaries = {}

nltk.download("punkt")

def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])

summaries["baseline"] = three_sentence_summary(sample_text)

set_seed(42)
pipe = pipeline("text-generation", model="gpt2-xl")
gpt2_query = sample_text + "\nTL;DR:\n"
pipe_out = pipe(gpt2_query, max_length=512, clean_up_tokenization_spaces=True)
summaries["gpt2"] = "\n".join(
    sent_tokenize(pipe_out[0]["generated_text"][len(gpt2_query) :]))

#hide_output
pipe = pipeline("summarization", model="t5-large")
pipe_out = pipe(sample_text)
summaries["t5"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))

#hide_output
pipe = pipeline("summarization", model="facebook/bart-large-cnn")
pipe_out = pipe(sample_text)
summaries["bart"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))

#hide_output
pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail")
pipe_out = pipe(sample_text)
summaries["pegasus"] = pipe_out[0]["summary_text"].replace(" .<n>", ".\n")

for model_name in summaries:
    print(model_name.upper())
    print(summaries[model_name])
    print("")

rouge_metric = load_metric("rouge")

reference = dataset["train"][1]["highlights"]
records = []
rouge_names = ["rouge1","rouge2","rougeL","rougeLsum"]

for model_name in summaries:
    rouge_metric.add(prediction=summaries[model_name], reference=reference)
    score = rouge_metric.compute()
    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
    records.append(rouge_dict)
print(pd.DataFrame.from_records(records, index=summaries.keys()))