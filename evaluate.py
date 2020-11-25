#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List
import numpy as np




def spearman_rank_correlation_annalysis(self,
        uniVocab: List[str],
        ngrams2ProbsA: np.array,
        ngrams2ProbsB: np.array):
    """ Analysis the Spearman Rank Correlation
    """
    rt = 0.

    allProbsA = []
    allProbsB = []
    for grams in ngrams2ProbsA:
        allProbsA.append(ngrams2ProbsA[grams])
        allProbsB.append(ngrams2ProbsB[grams])

    # sort
    argSortA = np.argsort(allProbsA)[::-1]
    argSortB = np.argsort(allProbsB)[::-1]
    # claculate
    rt = 1 - 6*(np.power((argSortA - argSortB), 2)).sum() / (np.power((len(argSortA),3)) - len(argSortA))
    return rt



