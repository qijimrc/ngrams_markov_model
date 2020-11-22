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
        tokens = self.vocab.bos + sent.strip().split() + self.vocab.eos
        tokIds = self.vocab.convert_tokens_to_ids(tokens)
        rt_prob = 1.
        # process the first `gramsNumber-1` tokens
        for i in range(1, gramsNumber-1):
            gramsIds = tokIds[:i+1]
            curTokId = tokIds[i]
            rt_prob *= self.vocab.ngramsProbsVocabs[1][curTokId] /\
                self.vocab.get_value(self.vocab.ngramsProbsVocabs[i+1], gramsIds)
        # process the rest
        for i in range(gramsNumber-1, len(sent)):
            gramsIds = tokIds[i-gramsNumber+1:i+1]
            curTokId = tokIds[i]
            rt_prob *= self.vocab.ngramsProbsVocabs[1][curTokId] /\
                self.vocab.get_value(self.vocab.ngramsProbsVocabs[i+1], gramsIds)
        rt_pp = np.power(rt_prob, -gramsNumber)
        return rt_prob, rt_pp


    def predict_topk_with_index(self,
            sent: str,
            gramsNumber: int,
            idx: int,
            topK: int) -> float:
        """ Calculate the probability of a sentence based on ngrams model.
        """ 
        assert idx >0
        tokens = self.vocab.bos + sent.strip().split() + self.vocab.eos
        tokIds = self.vocab.convert_tokens_to_ids(tokens)
        rt = 1.
        # process the first `gramsNumber-1` tokens
        for i in range(1, gramsNumber-1):
            gramsIds = tokIds[:i+1]
            curTokId = tokIds[i]
            rt *= self.vocab.ngramsProbsVocabs[1][curTokId] /\
                self.vocab.get_value(self.vocab.ngramsProbsVocabs[i+1], gramsIds)
            if i == topK:
                prevToksIds = tokIds[:i]
                allToksProbs = []
                for j in range(len(self.vocab.uniVocab)):
                    _curTok, _curTokId = self.vocab.uniVocab[j], j
                    allToksIds = prevToksIds + [_curTokId]
                    gramProb = self.vocab.get_value(self.vocab.ngramsProbsVocabs[i+1], allToksIds)
                    allToksProbs.append((_curTok ,self.vocab.ngramsProbsVocabs[1][_curTokId] / gramProb))
                allToksProbs = sorted(allToksProbs, lambda x,y:y, reverss=True)
                rankCurTok = [x[0] for x in allToksProbs].index(tokens[i])
                allToksProbs = allToksProbs[:topK]
                return allToksProbs, (tokens[i], rt, rankCurTok)
        # process the rest
        for i in range(gramsNumber-1, len(sent)):
            gramsIds = tokIds[i-gramsNumber+1:i+1]
            curTokId = tokIds[i]
            rt *= self.vocab.ngramsProbsVocabs[i][curTokId] / self.vocab.ngramsProbsVocabs[i+1][gramsIds]
            if i == topK:
                prevToksIds = tokIds[:i]
                allToksProbs = []
                for j in range(len(self.vocab.uniVocab)):
                    _curTok, _curTokId = self.vocab.uniVocab[j], j
                    allToksIds = prevToksIds + [_curTokId]
                    gramProb = self.vocab.get_value(self.vocab.ngramsProbsVocabs[i+1], allToksIds)
                    allToksProbs.append((_curTok ,self.vocab.ngramsProbsVocabs[1][_curTokId] / gramProb))
                allToksProbs = sorted(allToksProbs, lambda x,y:y, reverss=True)
                rankCurTok = [x[0] for x in allToksProbs].index(tokens[i])
                allToksProbs = allToksProbs[:topK]
                return allToksProbs, (tokens[i], rt, rankCurTok)



