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
import random
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

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epoch = 0
batch_size = 64
teacher_forcing_ratio = 0.9

'''
    sentence2Tensor: Get a list of sentences and output a tensor containing the indexed list.
                    The output's shape will be [max_seq_lens, batch_size]
'''


def sentence2Tensor(dict, input_list: List[List[str]], transpose, padding):
    sentence_indexes = []
    lens = []
    for sentence in input_list:
        indexes = []
        if padding == True:
            indexes.append(1)
        for word in sentence:
            if word in dict:
                indexes.append(dict[word])
            else:
                indexes.append(3)
        if padding == True:
            indexes.append(2)  # append </s>
        sentence_indexes.append(indexes)
        lens.append(indexes.__len__())

    max_len = max((len(l) for l in sentence_indexes))
    sentence_indexes = list(map(lambda l: l + [0] * (max_len - len(l)), sentence_indexes))

    if transpose == True:
        sentence_indexes = list(zip(*sentence_indexes))

    return torch.tensor(sentence_indexes, dtype=torch.long, device=device), max_len, lens


'''
    Attention:Attention part
'''


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = encoder_hidden_dim
        self.dec_hid_dim = dec_hid_dim

        self.attention = nn.Linear((encoder_hidden_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.value = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))

        energy = energy.permute(0, 2, 1)

        v = self.value.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


'''
    Encoder:Encoder class, encodes the source sentences and outputs hidden state.
'''


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()

        self.input_dim = input_vocab_size
        self.emb_dim = embed_dim
        self.enc_hid_dim = encoder_hidden_dim
        self.dec_hid_dim = decoder_hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_vocab_size, embed_dim)

        self.rnn = nn.GRU(embed_dim, encoder_hidden_dim, bidirectional=True)

        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        packed_outputs, hidden = self.rnn(embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden

'''
    Decoder: decoder class, gets a input and predicts.
'''
class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_dim, encoder_hidden_dim, decoder_hidden_dim, dropout, attention):
        super().__init__()

        self.emb_dim = embed_dim
        self.enc_hid_dim = encoder_hidden_dim
        self.dec_hid_dim = decoder_hidden_dim
        self.output_dim = output_vocab_size
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_vocab_size, embed_dim)

        self.rnn = nn.GRU((encoder_hidden_dim * 2) + embed_dim, decoder_hidden_dim)

        self.out = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim + embed_dim, output_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_outputs, mask)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)

        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        return output, hidden.squeeze(0), a.squeeze(1)


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, learning_rate=0.01):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.lr = learning_rate

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.attention = Attention(hidden_size * 2, hidden_size * 2)
        print("start to init encoder and decoder.")
        self.encoder = Encoder(vocab.src.__len__(), embed_size, hidden_size * 2, hidden_size * 2,
                               0.2).to(device)
        print("Encoder is done!")
        self.decoder = Decoder(vocab.tgt.__len__(), embed_size, hidden_size * 2, hidden_size * 2, 0.2,
                               self.attention).to(device)
        print("Decoder is done!")

    def create_mask(self, src):
        mask = (src != 0).permute(1, 0)
        return mask

    '''
        evaluate: gets a input sentence and output indexed prediction.
    '''

    def evaluate(self, src_sents: List[List[str]], tgt_sents: List[List[str]]):

        self.eval()
        beam_size = 10
        src_tensor, src_len, src_lens = sentence2Tensor(self.vocab.src, src_sents, True, True)

        tgt_tensor = torch.tensor(tgt_sents, device=device)
        batch_size = src_tensor.shape[1]
        max_len = src_tensor.shape[0]

        trg_vocab_size = self.decoder.output_dim
        mask = self.create_mask(src_tensor)

        encoder_outputs, hidden = self.encoder(src_tensor, src_lens)

        input = tgt_tensor[0, :]
        output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)
        topv, top1 = output.data.topk(beam_size)
        output_weight = []
        output_list = []
        hidden_list = []
        attention_list = []


        '''
            beam search and get the best prediction
        '''

        # get the first beam_size possible output
        for i in range(beam_size):
            __ = []
            __.append(top1[0, i].item())
            output_weight.append(topv[0, i].item())
            output_list.append(__)
            hidden_list.append(hidden)
            attention_list.append(attention)

        max = 100
        # predict according to each possible output
        for t in range(0, max):
            for i in range(0, beam_size):
                if output_list[i][output_list[i].__len__() - 1] != 2:
                    input = torch.tensor(output_list[i][output_list[i].__len__() - 1], device=device).view(1)
                    output, hidden_list[i], attention_list[i] = self.decoder(input, hidden_list[i], encoder_outputs,
                                                                             mask)
                    topv, top1 = output.data.topk(1)
                    output_weight[i] += topv.item()
                    output_list[i].append(top1.item())

        max_weight = -1
        max_idx = -1

        # get the best output
        for i in range(0, beam_size):
            if output_weight[i] / output_list[i].__len__() > max_weight:
                max_idx = i
                max_weight = output_weight[i] / output_list[i].__len__()

        return output_list[max_idx], output_weight[max_idx]

    '''
        train the model and update the encoder and decoder
    '''
    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]], batch_size) -> Tensor:

        self.train()
        optimizer = optim.Adam(self.parameters())

        # get source and target tensors
        src_tensor, src_len, src_lens = sentence2Tensor(self.vocab.src, src_sents, True, False)
        tgt_tensor, tgt_len, tgt_lens = sentence2Tensor(self.vocab.tgt, tgt_sents, True, False)

        batch_size = src_tensor.shape[1]
        max_len = tgt_tensor.shape[0]

        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)

        attentions = torch.zeros(max_len, batch_size, src_tensor.shape[0]).to(device)


        # encode source sentences
        encoder_outputs, hidden = self.encoder(src_tensor, src_lens)

        input = tgt_tensor[0, :]

        mask = self.create_mask(src_tensor)

        # train the decoder one word per time.
        for i in range(1, max_len):
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)

            outputs[i] = output
            attentions[i] = attention
            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = tgt_tensor[i] if teacher_force else top1

        outputs = outputs[1:].view(-1, outputs.shape[-1])
        tgt_tensor = tgt_tensor[1:].view(-1)

        loss = self.criterion(outputs, tgt_tensor)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)

        optimizer.step()

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

    '''
        beam search is implemented in NMT.evaluate()
    '''
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
        src_tensors = sentence2Tensor(self.vocab.src, strlist, True, False)
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
        torch.save(self, path)


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
                torch.save(model, "NMT.model")
                # torch.save(model.decoder, 'decoder.model')
                # torch.save(model.encoder, 'encoder.model')
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
    model = torch.load('NMT.model')
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')
    train_data = list(zip(train_data_src, train_data_tgt))
    hyp_list = []
    output_strs = []
    for src_sents, tgt_sents in batch_iter(train_data, batch_size=1, shuffle=True):
        target_list = [1]
        target_lists = []
        target_lists.append(target_list)

        predict_output, likelihood = model.evaluate(src_sents, target_lists, 1)

        list_output = []
        output_str = ""
        refer_str = ""

        for i in range(tgt_sents[0].__len__()):
            refer_str += tgt_sents[0][i]
            refer_str += " "
        for i in range(predict_output.__len__()):
            if predict_output[i] in model.vocab.tgt.id2word:
                if predict_output[i] != 2 and predict_output[i] != 1:
                    output_str += model.vocab.tgt.id2word[predict_output[i]]
                list_output.append(model.vocab.tgt.id2word[predict_output[i]])
            else:
                output_str += "<unk>"
                list_output.append(3)
            output_str += " "
        hyp = Hypothesis(list_output, likelihood)
        hyp_list.append(hyp)
        output_str += "\n"
        output_strs.append(output_str)
        print("\nreference:\n", refer_str, "\n-------------------------------")
        print(output_str)

    f = open("work_dir/decode.txt", 'a')
    for i in range(hyp_list.__len__()):
        f.write(output_strs[i])
    f.close()
    print("start to calculate bleu score!")
    print(compute_corpus_level_bleu_score(train_data_tgt, hyp_list))


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


def main():
    '''
    args = docopt(__doc__)
    decode(args)
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
