import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import spacy

# these classes differ in that it stores 2 past and one forward sentance around an extracted relations (weird g1,g2,gg1 variables added because can't index in the doc.sents objects...)

# note: relations are extracted for data with both coreference resolution and not, they are appended so you can expect duplicates


class extraction(object):
    def __init__(self, token_data, coref_data):
        self.tokens=token_data
        self.corefs=coref_data
        self.tokenKeys=[f for f in self.tokens.keys()]

    def list_entities(self, text, *args):  # store all entities. NOTE: Sometimes entities' names match closely -> use levenstein measure ???  too complex 
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
            coref_text=self.corefs[i]
            s=0
            g1=0
            g2=0
            gg1=0
            t=0
            cs=0
            cg1=0
            cg2=0
            cgg1=0
            ct=0
            for g in text.sents:
                s=s+1 
                g2=g1
                g1=t
                t=gg1
                gg1=g
                cs=s+1 
                cg2=g1
                cg1=t
                ct=gg1
                cgg1=g
                dict_={}
                org=self.find_org(t,ents)
                if s>=3:
                    if len(org)>=2:
                        SVO=self.check_synonym_SVO(t,*verbs_selec)
                        if SVO==True:
                            sbj=self.find_sbj(t,org)
                            if len(sbj)>0:
                                obj=self.find_obj(t,org)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation   # context don't work: not iterable 
                                    dict_["sentence"]=t
                                    dict_["sentence_context"]=g2+G1+t+GG1
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    dict_["other_type_context"]=cg2+cG1+ct+cGG1
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
                                        dict_["sentence_context"]=g2+G1+t+GG1
                                        dict_["entityA"]=obj
                                        dict_["entityB"]=sbj
                                        dict_["other_type_context"]=cg2+cG1+ct+cGG1
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
            text=self.corefs[i]
            ents=self.list_entities(text,*orgs_coref_tokens)
            s=0
            g1=0
            g2=0
            gg1=0
            t=0
            cs=0
            cg1=0
            cg2=0
            cgg1=0
            ct=0
            for g in text.sents:
                s=s+1 
                g2=g1
                g1=t
                t=gg1
                gg1=g
                cs=s+1 
                cg2=g1
                cg1=t
                ct=gg1
                cgg1=g
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if s>=3:
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
                                    dict_["sentence_context"]=g2+G1+t+GG1
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    dict_["other_type_context"]=cg2+cG1+ct+cGG1
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
                                        dict_["sentence_context"]=g2+G1+t+GG1
                                        dict_["entityA"]=obj
                                        dict_["entityB"]=sbj
                                        dict_["other_type_context"]=cg2+cG1+ct+cGG1
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

    def attribute_pattern(self, relation, verbs_selec, orgs_token, orgs_coref_tokens):
        tuples_token=[]
        tuples_coref=[]
        for i in self.tokenKeys:
            text=self.tokens[i]
            ents=self.list_entities(text,*orgs_token)
            coref_text=self.corefs[i]
            s=0
            g1=0
            g2=0
            gg1=0
            t=0
            cs=0
            cg1=0
            cg2=0
            cgg1=0
            ct=0
            for g in text.sents:
                s=s+1 
                g2=g1
                g1=t
                t=gg1
                gg1=g
                cs=s+1 
                cg2=g1
                cg1=t
                ct=gg1
                cgg1=g
                dict_={}
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if s>=3:
                    if len(org)>=2:
                        attr=self.check_synonym_attr(text.sents[s],*verbs_selec)
                        if attr==True:
                            sbj=self.find_sbj(text.sents[s],org)
                            if len(sbj)>0:
                                obj=self.find_obj_attr(text.sents[s],org)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=text.sents[s]
                                    dict_["sentence_context"]=g2+G1+t+GG1
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    dict_["other_type_context"]=cg2+cG1+ct+cGG1
                                    tuples_coref.append(dict_)

        for i in self.tokenKeys:
            token_text=self.tokens[i]
            text=self.corefs[i]
            ents=self.list_entities(text,*orgs_coref_tokens)
            s=0
            g1=0
            g2=0
            gg1=0
            t=0
            cs=0
            cg1=0
            cg2=0
            cgg1=0
            ct=0
            for g in text.sents:
                s=s+1 
                g2=g1
                g1=t
                t=gg1
                gg1=g
                cs=s+1 
                cg2=g1
                cg1=t
                ct=gg1
                cgg1=g
                dict_={}
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if s>=3:
                    if len(org)>=2:
                        attr=self.check_synonym_attr(text.sents[s],*verbs_selec)
                        if attr==True:
                            sbj=self.find_sbj(text.sents[s],org)
                            if len(sbj)>0:
                                obj=self.find_obj_attr(text.sents[s],org)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=text.sents[s]
                                    dict_["sentence_context"]=g2+G1+t+GG1
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    dict_["other_type_context"]=cg2+cG1+ct+cGG1
                                    tuples_coref.append(dict_)

        return tuples_token, tuples_coref



    def acl_pattern(self, relation, verbs_selec, orgs_token, orgs_coref_tokens):
        tuples_token=[]
        tuples_coref=[]
        for i in self.tokenKeys:
            text=self.tokens[i]
            ents=self.list_entities(text,*orgs_token)
            coref_text=self.corefs[i]
            s=0
            g1=0
            g2=0
            gg1=0
            t=0
            cs=0
            cg1=0
            cg2=0
            cgg1=0
            ct=0
            for g in text.sents:
                s=s+1 
                g2=g1
                g1=t
                t=gg1
                gg1=g
                cs=s+1 
                cg2=g1
                cg1=t
                ct=gg1
                cgg1=g
                dict_={}
                org=self.find_org(text.sents[s],ents)
                if s>=3:
                    if len(org)>=2:
                        acl=self.check_synonym_acl(text.sents[s],*verbs_selec)
                        if acl==True:
                            sbj=self.find_sbj(text.sents[s],org)
                            if len(sbj)>0:
                                obj=self.find_obj_acl(text.sents[s],org)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=text.sents[s]
                                    dict_["sentence_context"]=g2+G1+t+GG1
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    dict_["other_type_context"]=cg2+cG1+ct+cGG1
                                    tuples_coref.append(dict_)

        for i in self.tokenKeys:
            token_text=self.tokens[i]
            text=self.corefs[i]
            ents=self.list_entities(text,*orgs_coref_tokens)
            s=0
            g1=0
            g2=0
            gg1=0
            t=0
            cs=0
            cg1=0
            cg2=0
            cgg1=0
            ct=0
            for g in text.sents:
                s=s+1 
                g2=g1
                g1=t
                t=gg1
                gg1=g
                cs=s+1 
                cg2=g1
                cg1=t
                ct=gg1
                cgg1=g
                org=self.find_org(text.sents[s],ents)
                if s>=3:
                    if len(org)>=2:
                        acl=self.check_synonym_acl(text.sents[s],*verbs_selec)
                        if acl==True:
                            sbj=self.find_sbj(text.sents[s],org)
                            if len(sbj)>0:
                                obj=self.find_obj_acl(text.sents[s],org)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=text.sents[s]
                                    dict_["sentence_context"]=g2+G1+t+GG1
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    dict_["other_type_context"]=cg2+cG1+ct+cGG1
                                    tuples_coref.append(dict_)

        return tuples_token, tuples_coref
    
    def sentences_entities(self, coref, orgs):
        tuples_token=[]
        for i in self.tokenKeys:
            if coref==True:
                text=self.corefs[i]
            else:
                text=self.tokens[i]
            ents=self.list_entities(text,*orgs)    
            for t in text.sents:
                dict_={}
                org=self.find_org(t,ents)
                if len(org)>=2:
                    dict_["id"]=i
                    dict_["relation"]="entities_present"
                    dict_["sentence"]=t
                    dict_["entities"]=org
                    tuples_token.append(dict_)
        return tuples_token