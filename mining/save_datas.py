import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import spacy

class save_datas(object):
    def __init__(self, subdirectories, datadir, corefdirectory, model):
        self.datadir=datadir
        self.dir=corefdirectory
        self.model=model
        if subdirectories==True:
            self.sub=True
        else:
            self.sub=False

    def compute_token(self):
        if self.sub==True:
            directories=[]
            for file in os.listdir(self.datadir):
                d = os.path.join(self.datadir, file)
                if os.path.isdir(d):
                    directories.append(d)
            datas={}
            #spacy.require_cpu()
            nlp = spacy.load(self.model)
            for i in directories:
                pref=str(i.split('/')[-1])
                files = [f for f in listdir(i) if isfile(join(i, f))]
                for f in files:
                    if f[-4:]=='.txt':
                        #suf=f[:-4]
                        #name=pref+suf
                        with open(i+'/'+f, "r") as file:
                            data = file.read().replace("\n", "")
                        article = data.replace(u"\xa0", u" ")
                        datas[f]=nlp(article)
            return datas

        else:
            datas={}
            #spacy.require_cpu()
            nlp = spacy.load(self.model)
            files = [f for f in listdir(self.datadir) if isfile(join(self.datadir, f))]
            for f in files:
                #ff=f[:-4]
                with open(self.datadir+f, "r") as file:
                    data = file.read().replace("\n", "")
                article = data.replace(u"\xa0", u" ")
                datas[f]=nlp(article)
            return datas
