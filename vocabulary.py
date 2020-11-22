#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter, defaultdict
from typing import List
import copy


class Vocabulary:

    def __init__(self, N: int, corpus: List[str]):
        # predefine special tokens
        self.unk = "<UNK>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.gramsNumber = N
        # build vocabs
        self.uniVocab, self.ngramsFreqVocabs, self.ngramsProbsVocabs, \
            self.ngrams2Counts, self.count2Ngrams = self.build_ngrams_vocabs(N, corpus)


    def build_ngrams_vocabs(self, N: int, corpus: List[str]):
        """ Build an vocabulary with n-grams statistics

            Args
              - N: The number of n-grams.
              - corpus: A list of sentences with delimiter of balank space.
            Return
              - ngramsFreqVocabs: A dict for each key and value representing
                the grams number 'N' and it's counting matrix.
        """

        # statistics for all N-grams
        ngrams2Counts = defaultdict(Counter)
        for line in corpus:
            tokens = self.bos + line.strip().split() + self.eos
            for i in range(len(tokens)):
                # calculate
                j = i
                while i-j >= N-1 and j >= 0:
                    ngrams2Counts[i-j+1].update(tokens[j:i+1])
                    j -= 1

        # calculate ngrams count with freq
        count2Ngrams = defaultdict(lambda:defaultdict(set))
        for n in self.ngrams2Counts:
            n2C = copy.copy(self.ngrams2Counts[n])
            sortedN2C = sorted(n2C.items(), lambda k,v:v)
            # build
            for ngram, count in sortedN2C:
                count2Ngrams[n][count].add(ngram)

        # build vocab matrix
        sortedUnigrams = sorted(ngrams2Counts[1].items(), lambda x,y:y, reverse=True)
        uniVocab = [self.unk] + [k for k,v in sortedUnigrams]
        ngramsFreqVocabs = defaultdict(np.array)
        ngramsProbsVocabs = defaultdict(np.array)
        for n in ngrams2Counts:
            freqMatrix = np.zeros([len(uniVocab)]*n)
            # update
            for grams, count in ngrams2Counts[n].items():
                changeData = freqMatrix
                for k in range(len(grams)-1):
                    idx = uniVocab.index(grams[k])
                    changeData = changeData[idx]
                idx = uniVocab.index(grams[-1])
                changeData[idx] = count
            ngramsFreqVocabs[n] = freqMatrix
            ngramsProbsVocabs[n] = freqMatrix / freqMatrix.sum()

        return uniVocab, ngramsFreqVocabs, ngramsProbsVocabs, ngrams2Counts, count2Ngrams

    @classmethod
    def get_value(self, data: np.array, idxs: List[int]):
        if len(idxs) == 1:
            return data[idxs[0]]
        else:
            return self.get_value(data[idxs[0]], idxs[1:])

    def tune_with_Laplace_smoothing(self, B: int):
        """ Tune the all ngrams vocabularies with laplace smoothing (adding one) strategy.

            Args:
              - B: the parameter.
        """
        for n in self.ngramsFreqVocabs:
            totalN = sum(self.ngramsFreqVocabs[n].shape)
            self.ngramsProbsVocabs[n] = self.ngramsFreqVocabs+1/(totalN+B)

        

    def tune_with_held_out_smoothing(self, heldOutData: List[str]):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        def assgin(data, gramsNum, c2nInTrain, n2cInHeld):
            for i in range(len(data[gramsNum])):
                if len(data[gramsNum].shape) == 1:
                    r = data[gramsNum][i]
                    gramsSet = c2nInTrain[gramsNum][r]
                    N = len(n2cInHeld)
                    N_r = len(gramsSet)
                    T_r = sum([n2cInHeld[gramsNum][gram] for gram in gramsSet])
                    data[gramsNum][i] = T_r/(N_r*N)
                else:
                    assgin(data[gramsNum][i], gramsNum, c2nInTrain, n2cInHeld)

        # statistics for all N-grams on held-out data
        n2cInHeld = defaultdict(lambda: defaultdict(set))
        for line in heldOutData:
            tokens = self.bos + line.strip().split() + self.eos
            for i in range(len(tokens)):
                # count 1-grams
                n2cInHeld[1].update(tokens[i])
                # calculate >1 grams
                j = i-1
                while i-j >= self.gramsNumber-1 and j >= 0:
                    n2cInHeld[i-j+1].update(tokens[j:i+1])
        # tune
        assgin(self.ngramsProbsVocabs, self.gramsNum, self.count2Ngrams, n2cInHeld)
            

    def tune_with_cross_val_smoothing(self, heldOutData: List[str]):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        def assgin(data, gramsNum, c2nInTrain, n2cInTrain, c2nInHeld, n2cInHeld):
            for i in range(len(data[gramsNum])):
                if len(data[gramsNum].shape) == 1:
                    r = data[gramsNum][i]
                    gramsSet_1 = c2nInTrain[gramsNum][r]
                    gramsSet_2 = c2nInHeld[gramsNum][r]
                    N_1 = len(n2cInHeld)
                    N_r_1 = len(gramsSet_1)
                    T_r_1 = sum([n2cInHeld[gramsNum][gram] for gram in gramsSet_1])
                    N_2 = len(n2cInTrain)
                    N_r_2 = len(gramsSet_2)
                    T_r_2 = sum([n2cInTrain[gramsNum][gram] for gram in gramsSet_2])
                    data[gramsNum][i] = (T_r_1+T_r_2) / (N_1*N_r_1 + N_2+N_r_2)
                else:
                    assgin(data[gramsNum][i], gramsNum, c2nInTrain, n2cInTrain, c2nInHeld, n2cInHeld)

        # statistics for all N-grams on held-out data
        n2cInHeld = defaultdict(Counter)
        c2nInHeld = defaultdict(lambda: defaultdict(set))
        for line in heldOutData:
            tokens = self.bos + line.strip().split() + self.eos
            for i in range(len(tokens)):
                # calculate
                j = i
                while i-j >= self.gramsNumber-1 and j >= 0:
                    n2cInHeld[i-j+1].update(tokens[j:i+1])
                    j -= 1
        for n in n2cInHeld:
            n2C = copy.copy(self.ngrams2Counts[n])
            sortedN2C = sorted(n2C.items(), lambda k,v:v)
            # build
            for ngram, count in sortedN2C:
                c2nInHeld[n][count].add(ngram)
        # tune
        assgin(self.ngramsProbsVocabs, self.gramsNum, self.count2Ngrams, self.ngrams2Counts, c2nInHeld, n2cInHeld)



    def tune_with_good_turing_smoothing(self, ):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        def assgin(data, gramsNum, c2nInTrain):
            for i in range(len(data[gramsNum])):
                if len(data[gramsNum].shape) == 1:
                    r = data[gramsNum][i]
                    N_0 = len(c2nInTrain[gramsNum][0])
                    N_1 = len(c2nInTrain[gramsNum][1])
                    N_r = len(c2nInTrain[gramsNum][r])
                    N_r1 = len(c2nInTrain[gramsNum][r+1])
                    if r > 0:
                        data[gramsNum][i] = (r+1)*(N_r1/N_r)
                    else:
                        data[gramsNum][i] = N_0/N_1
                else:
                    assgin(data[gramsNum][i], gramsNum, c2nInTrain)
        # tune
        assgin(self.ngramsProbsVocabs, self.gramsNum, self.count2Ngrams)



    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.uniVocab.index(tk) if tk in self.uniVocab 
                else self.uniVocab.index(self.unk) for tk in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.uniVocab[idx] for idx in ids]

