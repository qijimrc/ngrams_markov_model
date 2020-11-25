#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter, defaultdict
from typing import List, Tuple
import copy
from tqdm import tqdm
import numpy as np


class Vocabulary:

    def __init__(self, N: int, corpus: List[str]=None):
        # predefine special tokens
        self.unk = "<UNK>"
        self.bos = "<BOS>"
        self.eos = "<EOS>"
        self.gramsNumber = N
        # build vocabs
        if corpus is not None:
            self.uniVocab, self.ngrams2Counts, self.count2Ngrams, self.ngrams2Probs\
                = self.build_ngrams_vocabs(N, corpus)


    def build_ngrams_vocabs(self, N: int, corpus: List[str]):
        """ Build an vocabulary with n-grams statistics

            Args
              - N: The number of n-grams.
              - corpus: A list of sentences with delimiter of balank space.
            Return
        """

        # statistics for all N-grams
        ngrams2Counts = defaultdict(Counter)
        for line in tqdm(corpus, desc="Build ngrams2Counts"):
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate
                j = i
                while i-j <= N-1 and j >= 0:
                    ngrams2Counts[i-j+1].update([tuple(tokens[j:i+1])])
                    j -= 1
        # process `unk`
        for n in range(1, N+1):
            ngrams2Counts[n][tuple([self.unk]*n)] = 0


        # calculate ngrams count with freq
        count2Ngrams = defaultdict(lambda:defaultdict(set))
        ngrams2Probs = defaultdict(lambda:defaultdict(float))
        for n in ngrams2Counts:
            # n2C = copy.copy(ngrams2Counts[n])
            # sortedN2C = sorted(n2C.items(), key=lambda x:x[1])
            # build
            total = sum(ngrams2Counts[n].values())
            for ngram in tqdm(ngrams2Counts[n], desc="Build count2Ngrams, ngrams2Probs for n="+str(n)):
                count = ngrams2Counts[n][ngram]
                ngrams2Probs[n][ngram] = ngrams2Counts[n][ngram] / total
                count2Ngrams[n][count].add(ngram)

        # build vocab matrix
        sortedUnigrams = sorted(ngrams2Counts[1].items(), key=lambda x:x[1], reverse=True)
        uniVocab = [self.unk] + [k[0] for k,v in sortedUnigrams]

        return uniVocab, ngrams2Counts, count2Ngrams, ngrams2Probs



    def tune_with_Laplace_smoothing(self, B: int):
        """ Tune the all ngrams vocabularies with laplace smoothing (adding one) strategy.

            Args:
              - B: the parameter.
        """
        for n in self.ngrams2Counts:
            totalN = sum(self.ngrams2Counts[n].values())
            for k in tqdm(self.ngrams2Probs[n], "Tuning with Laplace smoothing for n="+str(n)):
                self.ngrams2Probs[n][k] = (self.ngrams2Counts[n][k]+1) / (totalN+B)


    def tune_with_held_out_smoothing(self, heldOutData: List[str]):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """

        # statistics for all N-grams on held-out data
        n2cInHeld = defaultdict(Counter)
        for line in tqdm(heldOutData, desc="Processing with Held-Out Smoothing"):
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate >1 grams
                j = i
                while i-j <= self.gramsNumber-1 and j >= 0:
                    n2cInHeld[i-j+1].update([tuple(tokens[j:i+1])])
                    j -= 1
        # tune
        vocabLen = len(self.ngrams2Counts[1])
        for n in self.ngrams2Probs:
            N = sum(n2cInHeld[n].values())
            rt = defaultdict()
            total = sum(self.ngrams2Counts[n].values())
            for r in tqdm(self.count2Ngrams[n], desc="Tuning with Held-Out Smooth for n="+str(n)):
                if r == 0:
                    import ipdb
                    ipdb.set_trace()
                    if n == 1:
                        N_r =1
                    else:
                        N_r = np.power(vocabLen, n) - total
                    T_r = 0
                    for gramsH in n2cInHeld[n]:
                        if gramsH not in self.ngrams2Counts[n]:
                            import ipdb
                            ipdb.set_trace()
                            T_r += n2cInHeld[n][gramsH]
                else:
                    gramsSet = self.count2Ngrams[n][r]
                    N_r = len(gramsSet)
                    T_r = sum([n2cInHeld[n][g] for g in gramsSet])
                rt[r] = T_r/(N_r*N)

            for grams in tqdm(self.ngrams2Probs[n], desc="Tuning with Held-Out Smooth for n="+str(n)):
                r = self.ngrams2Counts[n][grams]
                newPorb = rt[r]
                self.ngrams2Probs[n][grams] = newPorb


            

    def tune_with_cross_val_smoothing(self, heldOutData: List[str]):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        # statistics for all N-grams on held-out data
        n2cInHeld = defaultdict(Counter)
        c2nInHeld = defaultdict(lambda: defaultdict(set))
        for line in heldOutData:
            tokens = [self.bos] + line.strip().split() + [self.eos]
            for i in range(len(tokens)):
                # calculate
                j = i
                while i-j >= self.gramsNumber-1 and j >= 0:
                    n2cInHeld[i-j+1].update([tuple(tokens[j:i+1])])
                    j -= 1
        for n in n2cInHeld:
            n2C = copy.copy(self.ngrams2Counts[n])
            sortedN2C = sorted(n2C.items(), lambda k,v:v)
            # build
            for ngram, count in sortedN2C:
                c2nInHeld[n][count].add(ngram)
        # tune
        for n in self.ngrams2Probs:
            for grams in tqdm(self.ngrams2Counts[n], desc="Tuning with cross-valid Smooth for n="+str(n)):
                r = self.ngrams2Counts[n][grams]
                N_r_1 = len(self.count2Ngrams[r])
                N_1 = sum(n2cInHeld[n].values())
                T_r_1 = 0
                for gramsH, gramsHC in n2cInHeld[n]:
                    if gramsH not in self.ngrams2Counts[n]:
                        T_r_1 += gramsHC

                N_r_2 = len(self.c2nInHeld[r])
                N_2 = sum(self.ngrams2Counts[n].values())
                T_r_2 = 0
                for gramsT, gramsTC in self.ngrams2Counts[n]:
                    if gramsT not in n2cInHeld[n]:
                        T_r_1 += gramsTC

                self.ngrams2Probs[n][grams] = (T_r_1+T_r_2) / (N_1*N_r_1 + N_2+N_r_2)



    def tune_with_good_turing_smoothing(self, ):
        """ Tune the all ngrams vocabularies with held-out strategy.

            Args:
              - B: the parameter.
        """
        for n in self.ngrams2Probs:
            for grams in tqdm(self.ngrams2Probs[n], desc="Tuning with Good-Turing Smooth for n="+str(n)):
                r = self.ngrams2Counts[n][grams]
                N_0 = self.count2Ngrams[n][0]
                N_1 = self.count2Ngrams[n][1]
                N_r = self.count2Ngrams[n][r]
                N_r1 = len(self.count2Ngrams[r+1])
                if r > 0:
                    self.ngrams2Probs[n][grams] = (r+1)*(N_r1/N_r)
                else:
                    self.ngrams2Probs[n][grams] = N_0/N_1


    def get_grams_probs(self, grams: Tuple):
        n = len(grams)
        if grams in self.ngrams2Probs[n]:
            return self.ngrams2Probs[n][grams]
        else:
            return self.ngrams2Probs[n][tuple([self.unk]*n)]

    def get_grams_candi_probs(self, tok: str, candiGrams: Tuple):
        n = len(candiGrams)
        if candiGrams not in self.ngrams2Probs[n]:
            jointProb = self.ngrams2Probs[n][tuple([self.unk]*n)]
        else: jointProb = self.ngrams2Probs[n][candiGrams]
        tokProb = 0.
        for grams in self.ngrams2Probs[n]:
            if grams[0] == tok:
                tokProb += self.ngrams2Probs[n][grams]
        return jointProb/tokProb

