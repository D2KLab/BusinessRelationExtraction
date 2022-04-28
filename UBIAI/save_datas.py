import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import spacy

class save_datas(object):
    def __init__(self, datadir, model):
        self.datadir=datadir
        self.model=model
            

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
