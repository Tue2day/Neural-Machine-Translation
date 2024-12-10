import numpy as np
from io import open
from collections import Counter
import re
import json
import jieba
from nltk.tokenize import word_tokenize


# Symbol(Begin of Sentence)(End of Sentence)
UNK_IDX = 0
PAD_IDX = 1

MAX_LENGTH = 20  # average length is 22.08897, choose 30 to be the truncate_length(GRU, not transformer)
MAX_WORDS = 20000


class Lang:
    def __init__(self, name):
        self.name = name
        # word frequency
        self.word2count = Counter()
        # Dictionary: word → index
        self.word2index = {}
        # Dictionary: index → word
        self.index2word = {}
        self.total_words = 2  # Initial n_words include "UNK" and "PAD"

    def count_word(self, sentence):
        for word in sentence:
            self.word2count[word] += 1

    def build_dict(self, max_words):
        ls = self.word2count.most_common(max_words)
        self.total_words = len(ls) + 2

        self.word2index = {w[0]: index + 2 for index, w in enumerate(ls)}
        # w[0]:word, w[1]:word frequency
        self.word2index['UNK'] = UNK_IDX
        self.word2index['PAD'] = PAD_IDX

        self.index2word = {v: k for k, v in self.word2index.items()}


def clean_zh(text):
    cleaned_text = re.sub(r"[^\u4e00-\u9fff0-9]", "", text)
    return cleaned_text


def clean_en(text):
    cleaned_text = re.sub(r'[^\w\s]+', ' ', text)
    cleaned_text = cleaned_text.replace('_', ' ')
    cleaned_text = cleaned_text.lower()
    return cleaned_text


def truncate_text(text):
    if len(text) >= MAX_LENGTH:
        text = text[:MAX_LENGTH - 1]
    return text


def tokenize_en(text):
    return word_tokenize(text)


def tokenize_zh(text):
    return list(jieba.cut(text))


def read_langs(language1, language2, file_name, reverse=False):
    print("Reading lines...")

    pairs = []

    # with open(file_name, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         data = json.loads(line)
    #         zh_tokenized = tokenize_zh(clean_zh(data['zh']))
    #         en_tokenized = tokenize_en(clean_en(data['en']))
    #         zh_text = ['BOS'] + truncate_text(zh_tokenized) + ['EOS']
    #         en_text = ['BOS'] + truncate_text(en_tokenized) + ['EOS']
    #
    #         pairs.append([zh_text, en_text])

    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.split('\t')
            zh_tokenized = tokenize_zh(clean_zh(line[1]))
            en_tokenized = tokenize_en(clean_en(line[0]))
            zh_text = ['BOS'] + truncate_text(zh_tokenized) + ['EOS']
            en_text = ['BOS'] + truncate_text(en_tokenized) + ['EOS']

            pairs.append([zh_text, en_text])

    # reverse the pair(change the translation direction)
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(language2)
        output_lang = Lang(language1)
    else:
        input_lang = Lang(language1)
        output_lang = Lang(language2)

    return input_lang, output_lang, pairs


def prepare_data(lang1, lang2, reverse=False):
    # train_file_name = './data/train_10k.jsonl'
    # val_file_name = './data/valid.jsonl'
    # test_file_name = './data/test.jsonl'
    train_file_name = '../data_short/nmt/en-cn/train.txt'
    val_file_name = '../data_short/nmt/en-cn/dev.txt'
    test_file_name = '../data_short/nmt/en-cn/test.txt'
    input_lang, output_lang, train_pairs = read_langs(lang1, lang2, train_file_name, reverse)
    _, _, valid_pairs = read_langs(lang1, lang2, val_file_name, reverse)
    _, _, test_pairs = read_langs(lang1, lang2, test_file_name, reverse)

    print("Read %s sentence pairs" % len(train_pairs))
    print("Counting words...")
    for pair in train_pairs:
        input_lang.count_word(pair[0])
        output_lang.count_word(pair[1])

    # Build word dictionary
    input_lang.build_dict(MAX_WORDS)
    output_lang.build_dict(MAX_WORDS)
    print("Counted words:")
    print(f"{input_lang.total_words} {input_lang.name} words")
    print(f"{output_lang.total_words} {output_lang.name} words")
    return input_lang, output_lang, train_pairs, valid_pairs, test_pairs


def encode(input_lang, output_lang, pairs, sort_by_len=True):
    for pair in pairs:
        pair[0] = [input_lang.word2index.get(word, 0) for word in pair[0]]
        pair[1] = [output_lang.word2index.get(word, 0) for word in pair[1]]

    # sort sentences by length
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x][0]))

    if sort_by_len:
        sorted_indices = len_argsort(pairs)
        pairs = [pairs[i] for i in sorted_indices]

    return pairs


input_lang, output_lang, train_pairs, val_pairs, test_pairs = prepare_data("Chinese", "English")
train_pairs = encode(input_lang, output_lang, train_pairs)
val_pairs = encode(input_lang, output_lang, val_pairs)
test_pairs = encode(input_lang, output_lang, test_pairs)


# Distribute sentence's index to every minibatch
def get_index_batch(n, batch_size, shuffle=True):
    idx_list = np.arange(0, n, batch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatch = []
    for idx in idx_list:
        minibatch.append(np.arange(idx, min(idx + batch_size, n)))
    return minibatch


# Get the information of each batch's sentence
def get_data_info(seqs):
    lengths = [len(seq) for seq in seqs]
    seq_nums = len(seqs)  # Number of sentences in the batch
    max_len = np.max(lengths)  # Max length of sentence in the batch

    x = np.zeros((seq_nums, max_len)).astype('int32')
    x_lengths = np.array(lengths).astype('int32')  # Original length of sentences in the batch

    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq

    return x, x_lengths


def gen_examples(pairs, batch_size):
    index_batch = get_index_batch(len(pairs), batch_size)
    data_total = []
    for batch in index_batch:
        input_sentences = [pairs[t][0] for t in batch]
        output_sentences = [pairs[t][1] for t in batch]
        b_input, b_input_len = get_data_info(input_sentences)
        b_output, b_output_len = get_data_info(output_sentences)
        data_total.append((b_input, b_input_len, b_output, b_output_len))
    # len(data_total) = n / batch_size
    # each item includes {a batch of input vectors, length of input sentences in a batch,
    #                     a batch of output vectors, length of output sentences in a batch}
    return data_total


batch_size = 64
train_data = gen_examples(train_pairs, batch_size)
val_data = gen_examples(val_pairs, batch_size)

# Choose Truncate MAX_LENGTH
# sum = 0
# max = 0
# for i in range(len(train_pairs)):
#     if len(train_pairs[i][0]) > max:
#         max = len(train_pairs[i][0])
#     sum += len(train_pairs[i][0])
#
# print(sum / len(train_pairs))
# print(max)
