from __future__ import unicode_literals
from six.moves import xrange
import numpy as np
import _pickle
# The difference is that a defaultdict will "default" a value if that key
# has not been set yet. If you didn't use a defaultdict you'd have to check
# to see if that key exists, and if it doesn't, set it to what you want.
from collections import defaultdict
import re
import pandas as pd
import os.path

classes = {
    'very negative' : 0,
    'negative' : 1,
    'neutral' : 2,
    'positive' : 3,
    'very positive' : 4
    }

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, embedding_dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * embedding_dim
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode(errors='ignore')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    # First column for UNKNOWN
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_train(filename='data/sst1/sst1_train.txt'):
    data = []
    x_train = []
    y_train = []
    with open(filename, 'r') as f:
        data.extend(f.readlines())
    for i, line in enumerate(data):
        if i % 2 == 0:
            x_train.append(line.rstrip())
        if i % 2 == 1:
            y_train.append(classes[line.rstrip()])
    return x_train, y_train

def load_dev(filename='data/sst1/sst1_dev.txt'):
    data = []
    x_dev = []
    y_dev = []
    with open(filename, 'r') as f:
        data.extend(f.readlines())
    for i, line in enumerate(data):
        if i % 2 == 0:
            x_dev.append(line.rstrip())
        if i % 2 == 1:
            y_dev.append(classes[line.rstrip()])
    return x_dev, y_dev


def build_data_cv(clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    x_train, y_train = load_train()
    x_dev, y_dev = load_dev()
    for i in range(len(x_train)):
        rev = []
        rev.append(x_train[i].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = \
            {"y":y_train[i],
             "text": orig_rev,
             "num_words": len(orig_rev.split()),
             "split": 1
            }
        revs.append(datum)
    # print (revs[0], revs[1])
    for i in range(len(x_dev)):
        rev = []
        rev.append(x_dev[i].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = \
            {"y":y_dev[i],
             "text": orig_rev,
             "num_words": len(orig_rev.split()),
             "split": 0
            }
        revs.append(datum)
    return revs, vocab

def clean_str(string, TREC=False):
    re.compile('(http|https):\/\/[^\/"]+[^ |"]*')
    string = re.sub('(http|https):\/\/[^\/"]+[^ |"]*', "http", string)
    match = re.search("(#\w+)", string)
    if match is not None:
        for i in match.groups():
            string += " " + i + " "
    string += (" " + "chr200" + " ") * (len([chr(ord(input_x)) for input_x in string if ord(input_x) > 130]) // 4)
    string = ''.join([chr(ord(input_x)) for input_x in string if ord(input_x) < 130]). \
                replace("\\\\", "").replace("&gt;", ">"). \
                replace("&lt;", "<").replace("&amp;", "&")
    string += (" " + "chr201" + " ") * (len(string.split()))
    string += (" " + "chr202" + " ") * string.count(":)")
    string += (" " + "chr203" + " ") * string.count("@")
    string += (" " + "chr204" + " ") * string.count("http")
    string += (" " + "chr205" + " ") * string.count(":(")
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def get_idx_from_sent(sent, word_idx_map, max_length, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_length + 2 * pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, max_length, num_classes, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train_data, test_data = [], []
    train_labels, test_labels = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_length, filter_h)
        label = [0] * num_classes
        label[rev["y"]] = 1
        if rev["split"] == 0:
            test_data.append(sent)
            test_labels.append(label)
        else:
            train_data.append(sent)
            train_labels.append(label)
    train_data = np.array(train_data, dtype="int")
    test_data = np.array(test_data, dtype="int")
    train_labels = np.asarray(train_labels, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.float32)

    return [train_data, train_labels, test_data, test_labels]

def create_data(revs, word_idx_map, max_length, num_classes):
    train_data, train_labels, test_data, test_labels = make_idx_data_cv(revs, word_idx_map, max_length, num_classes, 5)

    shuffle_indices = np.random.permutation(np.arange(len(train_data)))
    train_data = train_data[shuffle_indices]
    train_labels = train_labels[shuffle_indices]

    return train_data, train_labels, test_data, test_labels

def process_data(file_name):
    if os.path.isfile(file_name):
        print ("File {} already exists".format(file_name))
        return

    print ("Creating Dataset...")

    # load data
    revs, vocab = build_data_cv(clean_string=True)
    # print (pd.DataFrame(revs))
    max_length = np.max(pd.DataFrame(revs)["num_words"])
    print ("Data Loaded!")
    print ("Number of sentences: " + str(len(revs)))
    print ("Vocab size: " + str(len(vocab)))
    print ("Max sentence length: " + str(max_length))

    # load word2vec
    print ("Loading word2vec vectors..."),
    w2v_file = 'data/GoogleNews-vectors-negative300.bin'
    w2v = load_bin_vec(w2v_file, vocab)
    print ("Num words already in word2vec: " + str(len(w2v)))
    print ("Word2vec loaded!")

    #Addind random vectors for all unknown words
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)

    # In case we want to initialize with random word embeddings & not use word2vec
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    # dump to pickle file
    _pickle.dump([revs, W, W2, word_idx_map, vocab, max_length], open(file_name, "wb"))

    print ("Dataset Created!")

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffled_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffled_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            # min prevents end index from overfitting
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]