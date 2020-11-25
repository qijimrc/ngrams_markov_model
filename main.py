#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vocabulary import Vocabulary
from markov_model import MarkovModel
from evaluate import spearman_rank_correlation_annalysis
import copy
import pickle
import argparse
import os

def train(trainFile, devFile, gramsNumber,
        smoothStrategy, BLaplace):
    # process data
    with open(trainFile, "r") as f:
        corpusTrain = f.readlines()
    with open(devFile, "r") as f:
        corpusDev = f.readlines()
    corpusTrainDev = corpusTrain + corpusDev

    if smoothStrategy == "laplace":
        vocab = Vocabulary(gramsNumber, corpusTrainDev)
        vocab.tune_with_Laplace_smoothing(BLaplace)
    elif smoothStrategy == "held_out":
        vocab = Vocabulary(gramsNumber, corpusTrain)
        vocab.tune_with_held_out_smoothing(corpusDev)
    elif smoothStrategy == "cross_valid":
        vocab = Vocabulary(gramsNumber, corpusTrain)
        vocab.tune_with_cross_val_smoothing(corpusDev)
    elif smoothStrategy == "good_turing":
        vocab = Vocabulary(gramsNumber, corpusTrain)
        vocab.tune_with_good_turing_smoothing()
    else:
        raise KeyError
    return vocab

def predidct_file(model, gramsGumber, testFile, saveFile):
    # predict
    rt = []
    with open(testFile, "r") as f:
        for line in f:
            sentProb, sentPP = model.predict_sent_prob(line, gramsGumber)
            print(sentProb, sentPP)
            rt.append((sentProb, sentPP))
    # save results
    with open(saveFile, "w") as f:
        for sentProb, sentPP in rt:
            f.write(str(sentProb)+"\t"+str(sentPP)+"\n")

def predict_sent(model, gramsNumber, sent, idx, topK, saveFile):
    if os.path.isfile(sent):
        with open(sent, "r") as f:
            lines = f.readlines()
    else:
        lines = [sent]
    with open(saveFile, "a") as f:
        for line in lines:
            rt = model.predict_topk_with_index(line, gramsNumber, idx, topK)
            print(rt)
            # save results
            f.write("sent: " + line + "\n")
            f.write("top-k probs: ")
            f.write(" ".join(["({},{})".format(tok, str(prob)) for tok,prob in
                              rt[0]])+"\n")
            f.write("Current token: {}, with probs: {}, and ranking index:\
                    {}\n\n".format(rt[1][0], str(rt[1][1]), str(rt[1][2])))

def analysis_with_spearman_rank_correlation(allGrams, vocabA, vocabB, saveFile):
    rt = spearman_rank_correlation_annalysis(allGrams, vocabA, vocabB)
    with open(saveFile, "w") as f:
        f.write("The correlation result between two methods is {}".format(str(rt)))
    return rt




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default="pred_sent", help="which function will be used.")
    parser.add_argument("--train_file", type=str, default="data/s3/train.txt")
    parser.add_argument("--dev_file", type=str, default="data/s3/valid.txt")
    parser.add_argument("--test_file", type=str, default="data/s3/test.txt")
    parser.add_argument("--sent", type=str, default="日本 漫画 的 题材 非常 广泛")
    parser.add_argument("--idx", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--vocab_dir", type=str, default="outputs/vocabs")
    parser.add_argument("--out_dir", type=str, default="outputs/results")
    parser.add_argument("--grams_number", type=int, default=2)
    parser.add_argument("--smooth_strategy", type=str, default="laplace")
    parser.add_argument("--B_laplace", type=int, default=1)
    parser.add_argument("--analysis_model1", type=str, default="laplace")
    parser.add_argument("--analysis_model2", type=str, default="heldOut")
    args = parser.parse_args()

    vocab = train(args.train_file,
                args.dev_file,
                args.grams_number,
                args.smooth_strategy,
                args.B_laplace)
    model = MarkovModel(vocab)

    if args.func == "pred_file":
        predidct_file(model, args.grams_number, args.test_file,
                      os.path.join(args.out_dir,
                                   "test_file_results."+args.smooth_strategy+".txt"))
    elif args.func == "pred_sent":
        predict_sent(model, args.grams_number, args.sent, args.idx, args.topk,
                     os.path.join(args.out_dir, "test_sent_results."+args.smooth_strategy+".txt"))
    elif args.func == "spearman_analysis":
        vocab1 = train(args.train_file,
                      args.dev_file,
                      args.grams_number,
                      args.analysis_model1,
                      args.B_laplace)
        vocab2 = train(args.train_file,
                      args.dev_file,
                      args.grams_number,
                      args.analysis_model2,
                      args.B_laplace)
        analysis_with_spearman_rank_correlation(list(vocab1.ngrams2Counts.keys()),
                    vocab1,
                    vocab2,
                    os.path.join(args.out_dir,
                        "analysis_results."+args.analysis_model1+"-"+args.analysis_model2+".txt"))
    else:
        raise KeyError
