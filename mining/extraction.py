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

class extraction(object):
    def __init__(self, token_data, coref_data):
        self.tokens=token_data
        self.corefs=coref_data
        self.tokenKeys=[f for f in self.tokens.keys()]

    def list_entities(self, text, *args):  # store all entities. NOTE: Sometimes entities' names match closely -> use levenstein measure ???
        ents_=text.ents
        ents=[]
        for x in ents_:
            if x.text not in ents:
                for i in args:
                    if x.label_==i:
                        ents.append(x.text)
        return ents

    def find_org(self, s, ents): # iterates through tokens to find stored entities
        i=0
        ind=[]
        for token in s:
            if str(token) in ents:
                ind.append(i)
            i=i+1
        return ind

    def check_synonym_SVO(self, s, *list_): # checks whether the verbe (root in this context) is in the target list
        for token in s:
            if token.dep_=="ROOT":
                lem=token.lemma_
                if lem in list_:
                    return True
                else:
                    return False

    def find_sbj(self, s, ind): # detects subject in the sentence
        subjects=[]
        for i in ind:
            if s[i].dep_=="nsubj":
                subjects.append(s[i])
            elif s[i].head.dep_=="nsubj": # walk one step up the dependency tree
                subjects.append(s[i])
            else:
                continue # could move one more step up the dependency tree if we condition was came up before (excluding object and verbs)
        return subjects

    def find_nsubjpass(self, s, ind): # detects nsubjpass in sentence
        subjects=[]
        for i in ind:
            if s[i].dep_=="nsubjpass":
                subjects.append(s[i])
            elif s[i].head.dep_=="nsubjpass": # walk one step up the dependency tree
                subjects.append(s[i])
            else:
                continue # could move one more step up the dependency tree if we condition was came up before (excluding object and verbs)
        return subjects

    def find_obj(self, s, ind): # detects object in sentence
        subjects=[]
        for i in ind:
            if s[i].dep_=="dobj" or s[i].dep_=="pobj":
                subjects.append(s[i])
            elif s[i].head.dep_=="dobj" or s[i].head.dep_=="pobj": # walk one step up the dependency tree
                subjects.append(s[i])
            else:
                continue
        return subjects

    def check_synonym_attr(self, s, *list_): # checks whether the word (attr in this context) is in the target list
        for token in s:
            if token.dep_=="attr":
                lem=token.lemma_
                if lem in list_:
                    return True
                else:
                    return False

    def find_obj_attr(self, s, ind): # find the object in a "attribute" pattern sentense
        subjects=[]
        for i in ind:
            if s[i].dep_=="pobj":
                subjects.append(s[i])
            elif s[i].head.dep_=="pobj": # walk one step up the dependency tree
                subjects.append(s[i])
            else:
                continue
        return subjects

    def check_synonym_acl(self, s, *list_):
        for token in s:
            if token.dep_=="acl":
                lem=token.lemma_
                if lem in list_:
                    return True
                else:
                    return False

    def find_obj_acl(self, s, ind):
        subjects=[]
        for i in ind:
            if s[i].dep_=="pobj":
                subjects.append(s[i])
            elif s[i].head.dep_=="pobj": # walk one step up the dependency tree
                subjects.append(s[i])
            else:
                continue
        return subjects


    def extract_SVO(self, relation, verbs_selec, orgs_token, orgs_coref_tokens): # may try different combinations accross models
        tuples_token=[]
        tuples_coref=[]
        for i in self.tokenKeys:
            text=self.tokens[i]
            ents=self.list_entities(text,*orgs_token)
            coref_text=self.coref[i]
            for s in range(len(text.sents)):
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if len(org)>=2:
                    SVO=self.check_synonym_SVO(text.sents[s],*verbs_selec)
                    if SVO==True:
                        sbj=self.find_sbj(text.sents[s],org)
                        if len(sbj)>0:
                            obj=self.find_obj(text.sents[s],org)
                            if len(obj)>0:
                                dict_["id"]=i
                                dict_["relation"]=relation
                                dict_["sentence"]=text.sents[s]
                                dict_["sentence_context"]=text.sents[s-3]+text.sents[s-2]+text.sents[s-1]+text.sents[s]+text.sents[s+1]
                                dict_["entityA"]=sbj
                                dict_["entityB"]=obj
                                dict_["other_type_context"]=coref_text.sents[s-3]+coref_text.sents[s-2]+coref_text.sents[s-1]+coref_text.sents[s]+coref_text.sents[s+1]
                                tuples_token.append(dict_)
                            else:
                                continue
                        else:
                            nsubjpass=self.find_nsubjpass(text.sents[s],org)
                            if len(nsubjpass)>0:
                                obj=self.find_obj(text.sents[s],org)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=text.sents[s]
                                    dict_["sentence_context"]=text.sents[s-3]+text.sents[s-2]+text.sents[s-1]+text.sents[s]+text.sents[s+1]
                                    dict_["entityA"]=obj
                                    dict_["entityB"]=sbj
                                    dict_["other_type_context"]=coref_text.sents[s-3]+coref_text.sents[s-2]+coref_text.sents[s-1]+coref_text.sents[s]+coref_text.sents[s+1]
                                    tuples_token.append(dict_)
                                else:
                                    continue
                            else:
                                continue
                    else:
                        continue
                else:
                    continue
        for i in self.tokenKeys:
            token_text=self.tokens[i]
            text=self.coref[i]
            ents=self.list_entities(text,*orgs_coref_tokens)
            for s in range(len(text.sents)):
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if len(org)>=2:
                    self.check_synonym_SVO(text.sents[s],*verbs_selec)
                    if SVO==True:
                        sbj=self.find_sbj(text.sents[s],org)
                        if len(sbj)>0:
                            obj=self.find_obj(text.sents[s],org)
                            if len(obj)>0:
                                dict_["id"]=i
                                dict_["relation"]=relation
                                dict_["sentence"]=text.sents[s]
                                dict_["sentence_context"]=text.sents[s-3]+text.sents[s-2]+text.sents[s-1]+text.sents[s]+text.sents[s+1]
                                dict_["entityA"]=sbj
                                dict_["entityB"]=obj
                                dict_["other_type_context"]=token_text.sents[s-3]+token_text.sents[s-2]+token_text.sents[s-1]+token_text.sents[s]+token_text.sents[s+1]
                                tuples_coref.append(dict_)
                            else:
                                continue
                        else:
                            nsubjpass=self.find_nsubjpass(text.sents[s],org)
                            if len(nsubjpass)>0:
                                obj=self.find_obj(text.sents[s],org)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=text.sents[s]
                                    dict_["sentence_context"]=text.sents[s-3]+text.sents[s-2]+text.sents[s-1]+text.sents[s]+text.sents[s+1]
                                    dict_["entityA"]=obj
                                    dict_["entityB"]=sbj
                                    dict_["other_type_context"]=token_text.sents[s-3]+token_text.sents[s-2]+token_text.sents[s-1]+token_text.sents[s]+token_text.sents[s+1]
                                    tuples_coref.append(dict_)
                                else:
                                    continue
                            else:
                                continue
                    else:
                        continue
                else:
                    continue
        return tuples_token, tuples_coref

    def attribute_patern(self, relation, verbs_selec, orgs_token, orgs_coref_tokens):
        tuples_token=[]
        tuples_coref=[]
        for i in self.tokenKeys:
            text=self.tokens[i]
            ents=self.list_entities(text,*orgs_token)
            coref_text=self.coref[i]
            for s in range(len(text.sents)):
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if len(org)>=2:
                    self.check_synonym_attr(text.sents[s],*verbs_selec)
                    if attr==True:
                        sbj=self.find_sbj(text.sents[s],org)
                        if len(sbj)>0:
                            obj=self.find_obj_attr(text.sents[s],org)
                            if len(obj)>0:
                                dict_["id"]=i
                                dict_["relation"]=relation
                                dict_["sentence"]=text.sents[s]
                                dict_["sentence_context"]=text.sents[s-3]+text.sents[s-2]+text.sents[s-1]+text.sents[s]+text.sents[s+1]
                                dict_["entityA"]=sbj
                                dict_["entityB"]=obj
                                dict_["other_type_context"]=token_text.sents[s-3]+token_text.sents[s-2]+token_text.sents[s-1]+token_text.sents[s]+token_text.sents[s+1]
                                tuples_coref.append(dict_)

        for i in self.tokenKeys:
            token_text=self.tokens[i]
            text=self.coref[i]
            ents=self.list_entities(text,*orgs_coref_tokens)
            for s in range(len(text.sents)):
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if len(org)>=2:
                    self.check_synonym_attr(text.sents[s],*verbs_selec)
                    if attr==True:
                        sbj=self.find_sbj(text.sents[s],org)
                        if len(sbj)>0:
                            obj=self.find_obj_attr(text.sents[s],org)
                            if len(obj)>0:
                                dict_["id"]=i
                                dict_["relation"]=relation
                                dict_["sentence"]=text.sents[s]
                                dict_["sentence_context"]=text.sents[s-3]+text.sents[s-2]+text.sents[s-1]+text.sents[s]+text.sents[s+1]
                                dict_["entityA"]=sbj
                                dict_["entityB"]=obj
                                dict_["other_type_context"]=token_text.sents[s-3]+token_text.sents[s-2]+token_text.sents[s-1]+token_text.sents[s]+token_text.sents[s+1]
                                tuples_coref.append(dict_)

        return tuples_token, tuples_coref



    def acl_pattern(self, relation, verbs_selec, orgs_token, orgs_coref_tokens):
        tuples_token=[]
        tuples_coref=[]
        for i in self.tokenKeys:
            text=self.tokens[i]
            ents=self.list_entities(text,*orgs_token)
            coref_text=self.coref[i]
            for s in range(len(text.sents)):
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if len(org)>=2:
                    self.check_synonym_acl(text.sents[s],*verbs_selec)
                    if acl==True:
                        sbj=self.find_sbj(text.sents[s],org)
                        if len(sbj)>0:
                            obj=self.find_obj_acl(text.sents[s],org)
                            if len(obj)>0:
                                dict_["id"]=i
                                dict_["relation"]=relation
                                dict_["sentence"]=text.sents[s]
                                dict_["sentence_context"]=text.sents[s-3]+text.sents[s-2]+text.sents[s-1]+text.sents[s]+text.sents[s+1]
                                dict_["entityA"]=sbj
                                dict_["entityB"]=obj
                                dict_["other_type_context"]=token_text.sents[s-3]+token_text.sents[s-2]+token_text.sents[s-1]+token_text.sents[s]+token_text.sents[s+1]
                                tuples_coref.append(dict_)

        for i in self.tokenKeys:
            token_text=self.tokens[i]
            text=self.coref[i]
            ents=self.list_entities(text,*orgs_coref_tokens)
            for s in range(len(text.sents)):
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if len(org)>=2:
                    self.check_synonym_acl(text.sents[s],*verbs_selec)
                    if acl==True:
                        sbj=self.find_sbj(text.sents[s],org)
                        if len(sbj)>0:
                            obj=self.find_obj_acl(text.sents[s],org)
                            if len(obj)>0:
                                dict_["id"]=i
                                dict_["relation"]=relation
                                dict_["sentence"]=text.sents[s]
                                dict_["sentence_context"]=text.sents[s-3]+text.sents[s-2]+text.sents[s-1]+text.sents[s]+text.sents[s+1]
                                dict_["entityA"]=sbj
                                dict_["entityB"]=obj
                                dict_["other_type_context"]=token_text.sents[s-3]+token_text.sents[s-2]+token_text.sents[s-1]+token_text.sents[s]+token_text.sents[s+1]
                                tuples_coref.append(dict_)

        return tuples_token, tuples_coref
