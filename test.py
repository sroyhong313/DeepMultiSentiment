import tensorflow as tf
import numpy as np
import os
import time
import datetime
import utils
from model import TextCNN
import csv

# Parameters
# =========================================================

tf.flags.DEFINE_boolean("random",False,"Initialize with random word embeddings (default: False)")
tf.flags.DEFINE_boolean("static",False,"Keep the word embeddings static (default: False)")

tf.flags.DEFINE_integer("num_classes", 5, "Number of output classes (default: 5 (SST-1))")
tf.flags.DEFINE_integer("k-fold", 1, "Increase k-fold to 10 if no dev/test set is availabe (default:1)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of Character Embedding (default: 300 [Google W2Vec Dim])")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default:100)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.15, "Lambda value of L2-reg (default: 0.15)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (Default: 25)")
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("checkpoint_every", 300, "Save model after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print ("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print ("{}={}".format(attr.upper(), value))
print ("")

def load_test(filename='data/sst1/sst1_test.txt'):
    data = []
    x_test = []
    y_test = []
    with open(filename, "r") as f:
        data.extend(f.readlines())
    for i, line in enumerate(data):
        if i % 2 == 0:
            x_test.append(line.rstrip())
        else:
            y_test.append(line.rstrip())
    return x_test, y_test

def build_data(clean_string=True):
    revs = []
    vocab = defaultdict(float)
    x_test, y_test = load_test()
    for i in range(len(x_test)):
        rev = []
        rev.append(x_test[i].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = \
            {"y":y_test[i],
             "text": orig_rev,
             "num_words": len(orig_rev.split()),
             "split": 0
            }
        revs.append(datum)
    return revs, vocab