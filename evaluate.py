#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List
import numpy as np


def get_value(data: np.array, idxs: List[int]):
    if len(idxs) == 1:
        return data[idxs[0]]
    else:
        return get_value(data[idxs[0]], idxs[1:])


def permutations_all(N: int, A: List, cur: int, rt: List):
    if cur == N-1:
        rt.append(A)
    else:
        i = 0
        while i < N:
            A[cur] = i
            permutations_all(N, A, cur+1, rt)
            i += 1






def spearman_rank_correlation_annalysis(self,
        uniVocab: List[str],
        ngramsProbsVocabsA: np.array,
        ngramsProbsVocabsB: np.array,
        gramsNum: int):
    """ Analysis the Spearman Rank Correlation
    """
    rt = 0.
    assert ngramsProbsVocabsA.shape == ngramsProbsVocabsB.shape

    allGrams = []
    allProbsA = []
    allProbsB = []
    # get all indices permutations
    idxsAll = []
    permutations_all(gramsNum, [], 0, idxsAll)
    # get
    for idxs in idxsAll:
        allGrams.append([uniVocab[i] for i in idxs])
        allProbsA.append(get_value(ngramsProbsVocabsA, idxs))
        allProbsB.append(get_value(ngramsProbsVocabsB, idxs))
    # sort
    argSortA = np.argsort(allProbsA)[::-1]
    argSortB = np.argsort(allProbsB)[::-1]
    # claculate
    rt = 1 - 6*(np.power((argSortA - argSortB), 2)).sum() / (np.power((len(argSortA),3)) - len(argSortA))
    return rt



