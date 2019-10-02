# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry
import torch
from torch import *
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import re
import itertools

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
MAX_LENGTH = 55
max_epoch = 0
batch_size = 64


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def sentence_filter(string):
    print(string)
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    return string


def sentence2Tensor(dict, input_list: List[List[str]], transpose):
    total = input_list.__len__()
    sentence_indexes = []
    lens = []
    for sentence in input_list:
        # sentence = sentence_filter(sentence)
        indexes = []
        indexes.append(1)
        for word in sentence:
            if word in dict:
                indexes.append(dict[word])
            else:
                indexes.append(3)
        indexes.append(2)  # append </s>
        sentence_indexes.append(indexes)
        lens.append(indexes.__len__())

    #    batch = list(itertools.zip_longest(sentence_indexes, fillvalue=0))
    max_len = max((len(l) for l in sentence_indexes))
    sentence_indexes = list(map(lambda l: l + [0] * (max_len - len(l)), sentence_indexes))

    if transpose == True:
        sentence_indexes = list(zip(*sentence_indexes))

    return torch.tensor(sentence_indexes, dtype=torch.long, device=device), max_len, lens


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden,src_lens_list):
        #        embedded = self.embedding(input).view(MAX_LENGTH, batch_size, -1)
        embedded = self.embedding(input)
        batch_packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_lens_list,enforce_sorted=False)
        output = embedded
        output, hidden = self.gru(batch_packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_packed)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, batch_size,tgt_length, tgt_lens_list):
        '''

        input = input.unsqueeze(0)
        embeded = self.embedding(input)

        output, hidden = self.gru(embeded,hidden)

        return self.out(output.squeeze(0)),hidden

        '''

        output = self.embedding(input).view(tgt_length, batch_size, -1)
        batch_packed = torch.nn.utils.rnn.pack_padded_sequence(output, tgt_lens_list,enforce_sorted=False)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_packed)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        # output = self.softmax(self.out(output[0]))
        output = self.softmax(self.out(output))
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,
                 max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(MAX_LENGTH, batch_size, -1)
        embedded = self.dropout(embedded)
        tmp = embedded
        # for i in range(MAX_LENGTH):
        tmp = torch.cat((embedded[0], hidden[0]), 1)

        attn_weights = F.softmax(
            self.attn(tmp), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, hidden_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class NMT(object):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, learning_rate=0.1):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.lr = learning_rate

        print("start to init encoder and decoder.")
        self.encoder = EncoderRNN(vocab.src.__len__(), hidden_size).to(device)
        print("Encoder is done!")
        # self.decoder = AttnDecoderRNN(hidden_size, vocab.tgt.__len__(), dropout_rate).to(device)
        self.decoder = DecoderRNN(hidden_size, vocab.tgt.__len__()).to(device)
        print("Decoder is done!")
        # self.encoder = EncoderRNN()
        # set the model

        # initialize neural network layers...

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]], batch_size) -> Tensor:
        '''
        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.lr)

        criterion = nn.NLLLoss()
        # assert(src_sents.__len__(),tgt_sents.__len__())

        src_tensor, src_len = sentence2Tensor(self.vocab.src, src_sents,True)
        tgt_tensor, tgt_len = sentence2Tensor(self.vocab.tgt, tgt_sents,True)

        loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_hidden = self.encoder.initHidden()
        input_length = src_tensor.size(0)
        target_length = tgt_tensor.size(0)

        encoder_output, encoder_hidden = self.encoder(src_tensor, encoder_hidden)

'''

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.lr)

        criterion = nn.NLLLoss()
        # assert(src_sents.__len__(),tgt_sents.__len__())

        src_tensor, src_len, src_lens_list = sentence2Tensor(self.vocab.src, src_sents, True)
        tgt_tensor, tgt_len, tgt_lens_lits = sentence2Tensor(self.vocab.tgt, tgt_sents, True)

        #       for i in range(src_tensor_list.__len__()):
        loss = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_hidden = self.encoder.initHidden(batch_size)

        # encoder_outputs = torch.zeros(MAX_LENGTH, batch_size, self.encoder.hidden_size, device=device)

        encoder_output, encoder_hidden = self.encoder(src_tensor, encoder_hidden, src_lens_list)
        # encoder_outputs = encoder_output[0, 0]
        encoder_outputs = encoder_output

        decoder_hidden = encoder_hidden

        decoder_output, decoder_hidden = self.decoder(tgt_tensor, decoder_hidden, batch_size,tgt_len, tgt_lens_lits)
        # topv, topi = decoder_output.topk(1)
        # decoder_input = topi.squeeze().detach()

        #        for i in range(batch_size):
        #            tmp1 = decoder_output[i]
        #            tmp2 = tgt_tensor[i]

        decoder_output = decoder_output[0:].view(-1, decoder_output.shape[-1])
        tgt_tensor = tgt_tensor[0:].view(-1)
        loss = criterion(decoder_output, tgt_tensor)

        # for i in range(0, batch_size):
        #     tmp2 = torch.unsqueeze(decoder_output[i],0)
        #
        #     for j in range(0,MAX_LENGTH):
        #         tmp3 = tgt_tensor[i][j].view(1)
        #         tmp = criterion(tmp2, tgt_tensor[i][j].view(1))
        #         loss += tmp

        # if decoder_input.item() == 1:
        #     break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item()

    """
take a mini-batch of source and target sentences, compute the log-likelihood of 
target sentences.

Args:
    src_sents: list of source sentence tokens
    tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

Returns:
    scores: a variable/tensor of shape (batch_size, ) representing the 
        log-likelihood of generating the gold-standard target sentence for 
        each example in the input batch
"""

    # def encode(self, src_sents: List[List[str]]):
    #
    #
    #     """
    #     Use a GRU/LSTM to encode source sentences into hidden states
    #
    #     Args:
    #         src_sents: list of source sentence tokens
    #
    #     Returns:
    #         src_encodings: hidden states of tokens in source sentences, this could be a variable
    #             with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
    #         decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
    #     """

    # def decode(self, src_encodings: Tensor, decoder_init_state, tgt_sents: List[List[str]]):
    # """
    #         Given source encodings, compute the log-likelihood of predicting the gold-standard target
    #         sentence tokens
    #
    #         Args:
    #             src_encodings: hidden states of tokens in source sentences
    #             decoder_init_state: decoder GRU/LSTM's initial state
    #             tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`
    #
    #         Returns:
    #             scores: could be a variable of shape (batch_size, ) representing the
    #                 log-likelihood of generating the gold-standard target sentence for
    #                 each example in the input batch
    # """
    #     return 0

    #        return scores

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        strlist = [src_sent]
        src_tensors = sentence2Tensor(self.vocab.src, strlist, True)
        decoder_output, decoder_hidden = self.decoder()

        return strlist

    def evaluate_ppl(self, dev_data, batch_size: int = 64):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -self(src_sents, tgt_sents, src_sents.__len__())  # .sum()

            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """

        return torch.load(model_path)

    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(self, 'NMT.model')


#        torch.save(self.encoder, 'encoder.model')
#        torch.save(self.decoder, 'decoder.model')

# raise NotImplementedError()


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    max_epoch = int(args['--max-epoch'])
    batch_size = int(args['--batch-size'])

    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab,
                learning_rate=float(args['--lr-decay']))

    print("NMT init over!")
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    # tensors = sentence2Tensor(vocab.src,train_data_src)

    # print(tensors)
    while True:
        epoch += 1
        max_len = 0
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):

            train_iter += 1

            # (batch_size)
            loss = -model(src_sents, tgt_sents, src_sents.__len__())

            print(epoch, ':', loss)

            report_loss += loss
            cum_loss += loss

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += train_batch_size
            cumulative_examples += train_batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(
                                                                                             report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (
                                                                                                 time.time() - train_time),
                                                                                         time.time() - begin_time),
                      file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cumulative_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cumulative_tgt_words),
                                                                                             cumulative_examples),
                      file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data,
                                             batch_size=src_sents.__len__())  # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        # lr = lr * float(args['--lr-decay'])
                        # print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        # torch.save(model,model_save_path)

                        # print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

            if epoch == max_epoch:
                print('reached maximum number of epochs!', file=sys.stderr)
                torch.save(model.decoder, 'decoder.model')
                torch.save(model.encoder, 'encoder.model')
                exit(0)


'''

'''


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[
    List[Hypothesis]]:
    was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    print("test start")
'''
    
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')
'''

def predict(model,src_tensor,batch_size):
    with torch.no_grad():
        model.encoder.eval()
        model.decoder.eval()
        encoder_hidden = model.encoder.initHidden(batch_size)
        encoder_output, encoder_hidden = model.encoder(src_tensor,encoder_hidden)

        decoder_input_list = []
        for i in range(18):
            tmp = []
            if i == 0:
                tmp.append(1)
            else:
                tmp.append(0)
            decoder_input_list.append(tmp)

        decoder_input_tensor = torch.tensor(decoder_input_list,device=device)
        decoder_output, decoder_hidden = model.decoder(decoder_input_tensor, encoder_hidden,1,18)
        #topv, topi = decoder_output.data.topk(1)
        #    decoder_input = topi.squeeze().detach()
        return decoder_output

def main():
    '''
    model = torch.load('NMT.model')
    src_sent = ['wissen','sie', ',', 'eines', 'der', 'großen','<unk>', 'beim', 'reisen' ,'und', 'eine', 'der', 'freuden',
                'bei', 'der', '<unk>' ,'forschung' ,'ist']
    src_list = []
    src_list.append(src_sent)

    src_tensor,len = sentence2Tensor(model.vocab.src,src_list,True)
    predict_output = predict(model,src_tensor,1)

    print("over")
    '''

    args = docopt(__doc__)
    max_epoch = int(args['--max-epoch'])
    batch_size = int(args['--batch-size'])
    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
