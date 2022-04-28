import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import spacy
import neuralcoref

#check imports (otherwise, special file for coref), doc format coref, *args

class save_coref_datas(object):
    def __init__(self, datadir, corefdirectory, model, greedyness):
        self.datadir=datadir
        self.dir=corefdirectory
        self.model=model
        self.discr=greedyness
           

    def compute_token(self, subdir):
        if subdir==True:
            directories=[]
            for file in os.listdir(self.datadir):
                d = os.path.join(self.datadir, file)
                if os.path.isdir(d):
                    directories.append(d)
            datas={}
            nlp = spacy.load(self.model)
            for i in directories:
                files = [f for f in listdir(i) if isfile(join(i, f))]
                for f in files:
                    if f[-4:]=='.txt':
                        with open(i+'/'+f, "r") as file:
                            data = file.read().replace("\n", "")
                        article = data.replace(u"\xa0", u" ")
                        datas[i.split('/')[-1]+f]=nlp(article)
            return datas

        else:
            datas={}
            nlp = spacy.load(self.model)
            files = [f for f in listdir(self.datadir) if isfile(join(self.datadir, f))] # filter files
            for f in files:
                if f[-4:]=='.txt':
                    with open(self.datadir+f, "r") as file:
                        data = file.read().replace("\n", "")
                    article = data.replace(u"\xa0", u" ")
                    datas[f]=nlp(article)
            return datas

    def save_coref(self, tokens):
        keys = [f for f in tokens.keys()]
        datas={}
        nlp = spacy.load("en_core_web_sm")
        neuralcoref.add_to_pipe(nlp, greedyness=self.discr)
        for i in keys:
            datas[i]=nlp(str(tokens[i]))._.coref_resolved

        return datas

    def save_coref_without_compute_token(self,subdir):
        tokens=self.compute_token(subdir)
        return self.save_coref(tokens)

    def compute_sents_metrics(self,tokens,corefs):
        keys = [f for f in tokens.keys()]
        difference_metric=[] #stores the abs difference in number of sentences
        # corefs might not be doc object, debug later if necessary
        nlp_ = spacy.load("en_core_web_sm")
        for i in keys:
            t=0
            nn=nlp_(corefs[i])
            for i in nn.sents:
                t=t+1
            c=0
            #nnn=nlp_(tokens[i])
            for i in tokens[i].sents:
                c=c+1
            difference_metric.append(abs(t-c))

        return sum(difference_metric) / len(keys), len([i for i in difference_metric if i > 0])/len(keys)
        # prints average sentences mismatch overall
        # prints rate of article presenting a mismatch


    def coref_to_dir(self, tokens):  # method that prints to directory (and returns mismatch stats)
        keys = [f for f in tokens.keys()]
        datas={}
        nlp_ = spacy.load("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        neuralcoref.add_to_pipe(nlp, greedyness=self.discr)
        for i in keys:
            text=nlp(str(tokens[i]))._.coref_resolved
            with open(self.dir+i, mode = "x") as f:
                f.write(text)
                f.close
            t=0
            nn=nlp_(text)
            for i in nn.sents:
                t=t+1
            c=0
            #nnn=nlp_(tokens[i])
            for j in tokens[i].sents:
                c=c+1
            difference_metric.append(abs(t-c))
        return sum(difference_metric) / len(keys), len([i for i in difference_metric if i > 0])/len(keys)
