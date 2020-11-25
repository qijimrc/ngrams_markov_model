#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from vocabulary import Vocabulary
from typing import List


class MarkovModel:

    def __init__(self, vocabulary: Vocabulary):
        self.vocab = vocabulary


    def predict_sent_prob(self,
            sent: str,
            gramsNumber: int) -> float:
        """ Calculate the probability of a sentence based on ngrams model.
        """ 
        tokens = [self.vocab.bos] + sent.strip().split() + [self.vocab.eos]
        rt_prob = 1.
        # process the first `gramsNumber-1` tokens
        for i in range(1, min(gramsNumber-1, len(tokens))):
            grams = tuple(tokens[:i+1])
            prevGrams = grams[:-1]
            rt_prob *= self.vocab.get_grams_prob_n(grams) / self.vocab.get_grams_prob_n(prevGrams)
        # process the rest
        for i in range(gramsNumber-1, len(tokens)):
            grams = tuple(tokens[i-gramsNumber+1:i+1])
            prevGrams = grams[:-1]
            rt_prob *= self.vocab.get_grams_prob_n(grams) / self.vocab.get_grams_prob_n(prevGrams)
        rt_pp = np.power(rt_prob, -1/gramsNumber)
        return rt_prob, rt_pp


    def predict_topk_with_index(self,
            sent: str,
            gramsNumber: int,
            idx: int,
            topK: int) -> float:
        """ Calculate the probability of a sentence based on ngrams model.
        """ 
        assert idx >0
        tokens = [self.vocab.bos] + sent.strip().split() + [self.vocab.eos]
        rt = 1.
        # process the first `gramsNumber-1` tokens
        for i in range(1, min(gramsNumber-1, len(tokens))):
            grams = tuple(tokens[:i+1])
            prevGrams = grams[:-1]
            curTok = tuple([tokens[i]])
            rt *= self.vocab.get_grams_prob_n(grams) / self.vocab.get_grams_prob_n(prevGrams)
            # rt *= self.vocab.get_grams_candi_probs(tokens[i], grams)
            if i == idx:
                prevToks = grams[:-1]
                allToksProbs = []
                for j in range(len(self.vocab.uniVocab)):
                    _curTok = tuple([self.vocab.uniVocab[j]])
                    allToks = prevToks + _curTok
                    prob = self.vocab.get_grams_prob_n(allToks) / self.vocab.get_grams_prob_n(prevToks)
                    allToksProbs.append((_curTok ,prob))
                allToksProbs = sorted(allToksProbs, key=lambda x:x[1], reverse=True)
                allToksToks = [x[0] for x in allToksProbs]
                if curTok not in allToksToks:
                    rankCurTok = -1
                else:
                    rankCurTok = allToksToks.index(curTok)
                allToksProbs = allToksProbs[:topK]
                return allToksProbs, (tokens[i], rt, rankCurTok)
        # process the rest
        for i in range(gramsNumber-1, len(tokens)):
            grams = tuple(tokens[i-gramsNumber+1:i+1])
            curTok = tuple([tokens[i]])
            rt *= self.vocab.get_grams_prob_n(grams) /\
                self.vocab.get_grams_prob_n(curTok)
            # rt *= self.vocab.get_grams_candi_probs(tokens[i], grams)
            if i == idx:
                prevToks = grams[:-1]
                allToksProbs = []
                for j in range(len(self.vocab.uniVocab)):
                    _curTok = tuple([self.vocab.uniVocab[j]])
                    allToks = prevToks + _curTok
                    prob = self.vocab.get_grams_prob_n(allToks) / self.vocab.get_grams_prob_n(prevToks)
                    allToksProbs.append((_curTok ,prob))
                allToksProbs = sorted(allToksProbs, key=lambda x:x[1], reverse=True)
                allToksToks = [x[0] for x in allToksProbs]
                if curTok not in allToksToks:
                    rankCurTok = -1
                else:
                    rankCurTok = allToksToks.index(curTok)
                allToksProbs = allToksProbs[:topK]
                return allToksProbs, (tokens[i], rt, rankCurTok)



