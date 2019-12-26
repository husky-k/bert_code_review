import os
import sys
import json
import nltk
import random
import logging
import tensorflow as tf
import sentencepiece as spm

from glob import glob
from tensorflow.keras.utils import Progbar

sys.path.append("bert")

from bert import modeling, optimization, tokenization
from bert.run_pretraining import input_fn_builder, model_fn_builder

# configure logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s :  %(message)s')
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
log.handlers = [sh]

USE_TPU = False


regex_tokenizer = nltk.RegexpTokenizer("\w+")

def normalize_text(text):
  # lowercase text
  text = str(text).lower()
  # remove non-UTF
  text = text.encode("utf-8", "ignore").decode()
  # remove punktuation symbols
  text = " ".join(regex_tokenizer.tokenize(text))
  return text

def count_lines(filename):
  count = 0
  with open(filename) as fi:
    for line in fi:
      count += 1
  return count

print('text normalized: ' + normalize_text('Thanks to the advance, they have succeeded in getting over their adversaries.'))
