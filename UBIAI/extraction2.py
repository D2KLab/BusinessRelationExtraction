import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import spacy

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
    
    def find_sbj(self, s, ind, n): # detects subject in the sentence, goes up to N times into dependency tree 
        subjects=[]
        for i in ind:
            if s[i].dep_=="nsubj":
                subjects.append(s[i])
                ss=s[i].head
            for nn in range(2,n):
                if ss.dep_=="nsubj": # walk one step up the dependency tree
                    subjects.append(s[i])
                    ss=ss.head
        return subjects

    def find_nsubjpass(self, s, ind, n): # detects nsubjpass in sentence
        subjects=[]
        for i in ind:
            if s[i].dep_=="nsubjpass":
                subjects.append(s[i])
                ss=s[i].head
            for nn in range(2,n):
                if s[i].head.dep_=="nsubjpass": # walk one step up the dependency tree
                    subjects.append(s[i])
                    ss=ss.head
            else:
                continue # could move one more step up the dependency tree if we condition was came up before (excluding object and verbs)
        return subjects

    def find_obj(self, s, ind, n): # detects object in sentence
        subjects=[]
        for i in ind:
            if s[i].dep_=="dobj" or s[i].dep_=="pobj":
                subjects.append(s[i])
                ss=s[i].head
            for nn in range(2,n):
                if s[i].head.dep_=="dobj" or s[i].head.dep_=="pobj": # walk one step up the dependency tree
                    subjects.append(s[i])
                    ss=ss.head
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

    def find_obj_attr(self, s, ind, n): # find the object in a "attribute" pattern sentense
        subjects=[]
        for i in ind:
            if s[i].dep_=="pobj":
                subjects.append(s[i])
                ss=s[i].head
            for nn in range(2,n):
                if s[i].head.dep_=="pobj": # walk one step up the dependency tree
                    subjects.append(s[i])
                    ss=ss.head
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
                ss=s[i].head
            for nn in range(2,n):
                if s[i].head.dep_=="pobj": # walk one step up the dependency tree
                    subjects.append(s[i])
                    ss=ss.head
            else:
                continue
        return subjects


    def extract_2entitiesSVO(self, relation, token, coref, verbs_selec, orgs_token, orgs_coref_tokens, subj_n, obj_n): # may try different combinations accross models
        tuples_token=[]
        tuples_coref=[]
        if token==True:
            for i in self.tokenKeys:
                text=self.tokens[i]
                ents=self.list_entities(text,*orgs_token)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=2:
                        SVO=self.check_synonym_SVO(t,*verbs_selec)
                        if SVO==True:
                            sbj=self.find_sbj(t,org,subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj(t,org, obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation   # context don't work: not iterable 
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_token.append(dict_)
                                else:
                                    continue
                            else:
                                nsubjpass=self.find_nsubjpass(t,org, subj_n)
                                if len(nsubjpass)>0:
                                    obj=self.find_obj(t,org, obj_n)
                                    if len(obj)>0:
                                        dict_["id"]=i
                                        dict_["relation"]=relation
                                        dict_["sentence"]=t
                                        dict_["entityA"]=obj
                                        dict_["entityB"]=sbj
                                        tuples_token.append(dict_)
                                    else:
                                        continue
                                else:
                                    continue
                        else:
                            continue
                    else:
                        continue
        if coref==True:
            for i in self.tokenKeys:
                token_text=self.tokens[i]
                text=self.corefs[i]
                ents=self.list_entities(text,*orgs_coref_tokens)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=2:
                        self.check_synonym_SVO(t,*verbs_selec)
                        if SVO==True:
                            sbj=self.find_sbj(t,org, subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj(t,org, obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)
                                else:
                                    continue
                            else:
                                nsubjpass=self.find_nsubjpass(t,org, subj_n)
                                if len(nsubjpass)>0:
                                    obj=self.find_obj(t,org, obj_n)
                                    if len(obj)>0:
                                        dict_["id"]=i
                                        dict_["relation"]=relation
                                        dict_["sentence"]=t
                                        dict_["entityA"]=obj
                                        dict_["entityB"]=sbj
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

    def attribute_2entitiespattern(self, relation, verbs_selec, orgs_token, orgs_coref_tokens, subj_n, obj_n):
        tuples_token=[]
        tuples_coref=[]
        if token==True:
            for i in self.tokenKeys:
                text=self.tokens[i]
                ents=self.list_entities(text,*orgs_token)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=2:
                        attr=self.check_synonym_attr(t,*verbs_selec)
                        if attr==True:
                            sbj=self.find_sbj(t,org, subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj_attr(t,org, obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)
        if coref==True:
            for i in self.tokenKeys:
                token_text=self.tokens[i]
                text=self.corefs[i]
                ents=self.list_entities(text,*orgs_coref_tokens)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=2:
                        attr=self.check_synonym_attr(t,*verbs_selec)
                        if attr==True:
                            sbj=self.find_sbj(t,org, subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj_attr(t,org, obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)

        return tuples_token, tuples_coref



    def acl_2entitiespattern(self, relation, toke, coref, verbs_selec, orgs_token, orgs_coref_tokens,subj_n,obj_n):
        tuples_token=[]
        tuples_coref=[]
        if token==True:
            for i in self.tokenKeys:
                text=self.tokens[i]
                ents=self.list_entities(text,*orgs_token)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=2:
                        acl=self.check_synonym_acl(t,*verbs_selec)
                        if acl==True:
                            sbj=self.find_sbj(t,org,subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj_acl(t,org,obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)
        if coref==True:
            for i in self.tokenKeys:
                token_text=self.tokens[i]
                text=self.corefs[i]
                ents=self.list_entities(text,*orgs_coref_tokens)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=2:
                        acl=self.check_synonym_acl(t,*verbs_selec)
                        if acl==True:
                            sbj=self.find_sbj(t,org,subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj_acl(t,org,obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)

        return tuples_token, tuples_coref
    
    def sentences_2entities(self, coref, orgs):
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
                    tuples_token.append(dict_)
        return tuples_token
    
    def sentences_entity_products(self, coref):   # org + products
        tuples_token=[]
        for i in self.tokenKeys:
            if coref==True:
                text=self.corefs[i]
            else:
                text=self.tokens[i]
            org=self.list_entities(text,"ORG")
            prod=self.list_entities(text,"PRODUCT")
            for t in text.sents:
                dict_={}   
                org_=self.find_org(t,org)
                prod_=self.find_org(t,prod)
                if len(org_)>=1 & len(prod_)>=1:
                    dict_["id"]=i
                    dict_["relation"]="entities-products_present"
                    dict_["sentence"]=t
                    tuples_token.append(dict_)                    
        return tuples_token
        
    def extract_1entitiesSVO(self, relation, token, coref, verbs_selec, orgs_token, orgs_coref_tokens, subj_n, obj_n): # may try different combinations accross models
        tuples_token=[]
        tuples_coref=[]
        if token==True:
            for i in self.tokenKeys:
                text=self.tokens[i]
                ents=self.list_entities(text,*orgs_token)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=1:
                        SVO=self.check_synonym_SVO(t,*verbs_selec)
                        if SVO==True:
                            sbj=self.find_sbj(t,org,subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj(t,org, obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation  
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_token.append(dict_)
                                else:
                                    continue
                            else:
                                nsubjpass=self.find_nsubjpass(t,org, subj_n)
                                if len(nsubjpass)>0:
                                    obj=self.find_obj(t,org, obj_n)
                                    if len(obj)>0:
                                        dict_["id"]=i
                                        dict_["relation"]=relation
                                        dict_["sentence"]=t
                                        dict_["entityA"]=obj
                                        dict_["entityB"]=sbj
                                        tuples_token.append(dict_)
                                    else:
                                        continue
                                else:
                                    continue
                        else:
                            continue
                    else:
                        continue
        if coref==True:
            for i in self.tokenKeys:
                token_text=self.tokens[i]
                text=self.corefs[i]
                ents=self.list_entities(text,*orgs_coref_tokens)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=1:
                        self.check_synonym_SVO(t,*verbs_selec)
                        if SVO==True:
                            sbj=self.find_sbj(t,org, subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj(t,org, obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)
                                else:
                                    continue
                            else:
                                nsubjpass=self.find_nsubjpass(t,org, subj_n)
                                if len(nsubjpass)>0:
                                    obj=self.find_obj(t,org, obj_n)
                                    if len(obj)>0:
                                        dict_["id"]=i
                                        dict_["relation"]=relation
                                        dict_["sentence"]=t
                                        dict_["entityA"]=obj
                                        dict_["entityB"]=sbj
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
    
    def attribute_1entitiespattern(self, relation, verbs_selec, orgs_token, orgs_coref_tokens, subj_n, obj_n):
        tuples_token=[]
        tuples_coref=[]
        if token==True:
            for i in self.tokenKeys:
                text=self.tokens[i]
                ents=self.list_entities(text,*orgs_token)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=1:
                        attr=self.check_synonym_attr(t,*verbs_selec)
                        if attr==True:
                            sbj=self.find_sbj(t,org, subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj_attr(t,org, obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)
        if coref==True:
            for i in self.tokenKeys:
                token_text=self.tokens[i]
                text=self.corefs[i]
                ents=self.list_entities(text,*orgs_coref_tokens)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=1:
                        attr=self.check_synonym_attr(t,*verbs_selec)
                        if attr==True:
                            sbj=self.find_sbj(t,org, subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj_attr(t,org, obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)

        return tuples_token, tuples_coref  
    
    
    def acl_2entitiespattern(self, relation, toke, coref, verbs_selec, orgs_token, orgs_coref_tokens,subj_n,obj_n):
        tuples_token=[]
        tuples_coref=[]
        if token==True:
            for i in self.tokenKeys:
                text=self.tokens[i]
                ents=self.list_entities(text,*orgs_token)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=1:
                        acl=self.check_synonym_acl(t,*verbs_selec)
                        if acl==True:
                            sbj=self.find_sbj(t,org,subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj_acl(t,org,obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)
        if coref==True:
            for i in self.tokenKeys:
                token_text=self.tokens[i]
                text=self.corefs[i]
                ents=self.list_entities(text,*orgs_coref_tokens)
                for t in text.sents:
                    dict_={}
                    org=self.find_org(t,ents)
                    if len(org)>=1:
                        acl=self.check_synonym_acl(t,*verbs_selec)
                        if acl==True:
                            sbj=self.find_sbj(t,org,subj_n)
                            if len(sbj)>0:
                                obj=self.find_obj_acl(t,org,obj_n)
                                if len(obj)>0:
                                    dict_["id"]=i
                                    dict_["relation"]=relation
                                    dict_["sentence"]=t
                                    dict_["entityA"]=sbj
                                    dict_["entityB"]=obj
                                    tuples_coref.append(dict_)

        return tuples_token, tuples_coref
    
    # create class that finds two entities in subject or object -> read through example to specify what to write
    # attr move up dependency tree by 1?
        