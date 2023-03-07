from transformers import TapasTokenizer
import pandas as pd
import torch
import numpy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    AutoTokenizer,
    ElectraForQuestionAnswering,
)

from transformers import(
    BertConfig,
    AutoTokenizer,
    BertForQuestionAnswering,
    BertTokenizer,
    BertTokenizerFast,
    PreTrainedTokenizerFast,
    AutoModelWithLMHead,
    AutoModelForQuestionAnswering

)

HUGGINGFACE_MODEL_PATH = "/home/train_1/projects/MRC/Table_MRC/table_bert"
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
model= AutoModelForQuestionAnswering.from_pretrained(HUGGINGFACE_MODEL_PATH)
model.resize_token_embeddings(len(tokenizer))
import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


def update(question, file):
    query = question
    df_1 = file
    df = file.to_html()
    context = df.replace("\n", "")
    encodings = tokenizer(query, context, 
                      max_length=len(context),
                      truncation=False,
                      padding="max_length",
                      return_offsets_mapping=True,
                      )
    encodings = {key: torch.tensor([val]) for key, val in encodings.items()}             

    # Predict
    pred = model(encodings["input_ids"], attention_mask=encodings["attention_mask"])
    start_logits, end_logits = pred.start_logits, pred.end_logits
    
    token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
    pred_ids = encodings["input_ids"][0][token_start_index: token_end_index + 1]
    answer_text = tokenizer.decode(pred_ids)
    answer_text = cleanhtml(answer_text)

    # Offset
    answer_start_offset = int(encodings['offset_mapping'][0][token_start_index][0][0])
    answer_end_offset = int(encodings['offset_mapping'][0][token_end_index][0][1])
    answer_offset = (answer_start_offset, answer_end_offset)

    def highlight_cell(val):
      if answer_text in str(val):
          return 'background-color: green'
      else:
          return print(str(val))

    ab = df_1.style.applymap(highlight_cell)

    return answer_text, ab
