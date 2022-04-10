import shap
import transformers
import torch
import numpy as np
import scipy as sp
import pandas as pd
#########################################
# load data
########################################
from datasets import load_dataset
dataset = load_dataset('text', data_files='text_example.txt', )
for i in range(5):
    print("text item {}: ".format(i))
    print(dataset['train']['text'][i])

from transformers import AutoTokenizer, AutoModel, pipeline

#########################################
# build pipeline
########################################

# how to use?
print('building pipeline ...')
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
      return_all_scores = True,
)
print('checking pipeline: ')
for sentence in dataset['train']['text'][0:5]:
  print(sentence, sentiment_analysis(sentence), sep='\n')
