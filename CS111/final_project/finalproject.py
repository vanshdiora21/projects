#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:11:48 2022

@author: vansh
"""
import math
def clean_text(txt):
    """removes punctuation marks from a text"""
    txt = txt.lower()
    for symbol in """.,?"'!;:""":
        txt = txt.replace(symbol,"")
    txt = txt.split()
    return txt
def stem(s):
    """accepts a string as a parameter. 
    The function should then return the stem of s"""
    if s[-3:] == "ing":
        s = s[:-3]
    elif s[-2:] == "er":
        s = s[:-2]
    elif s[-3:] == "ity":
        s = s[:-3]
    elif s[-4:] == "ness":
        s = s[:-4]
    elif s[-3:] == "ery":
        if len(s) == 4:
            s = s
        else:
            s = s[:-3]
    elif s[:2] == "re":
        s = s[2:]
    elif s[:4] == "anti":
        s = s[4:]
    elif s[-1] == "y":
        s = s[:-1] + "i"
    elif s[-2:] == "es":
        s = s[:-2]
    elif s[:4] == "over":
        s = s[4:]
    elif s[:3] == "dis":
        s = s[3:]
    elif s[:-3] == "ies":
        s = s[:-3] + "i"  
    elif s[-1] == "s":
        s = s[:-1]
    return s
    
def compare_dictionaries(d1,d2):
    """two feature dictionaries d1 and d2 as inputs, 
    and it should compute and return their log 
    similarity score"""
    if d1 == {}:
        return -50
    else:
        score = 0
        total = 0
        for x in d1:
            total += d1[x]
        for y in d2:
            if y in d1:
                score +=  d2[y] * math.log(d1[y]/total)
            else:
                score +=  d2[y] * math.log(0.5 / total)
        return score
class TextModel:
    """model, analyze, and score the similarity of text samples"""
    def __init__(self,model_name):
        """constructs a new TextModel object by accepting a string model_name as a parameter """
        self.name = model_name
        self.words = {}
        self.word_lengths = {}
        self.stems = {}
        self.sentence_lengths = {}
        self.punctuation = {}
        
        
    def __repr__(self):
        """Return a string representation of the TextModel."""
        s = ""
        s+= "text model name: " + self.name + "\n"
        s+= "  number of words: " + str(len(self.words)) + "\n"
        s += "  number of word lengths: " + str(len(self.word_lengths)) + "\n"
        s += "  number of stems: " + str(len(self.stems)) + "\n"
        s += "  number of sentence lengths: " + str(len(self.sentence_lengths)) + "\n"
        s += "  number of punctuations: " + str(len(self.punctuation))
        return s
    def add_string(self,s):
        """Analyzes the string txt and adds its pieces
        to all of the dictionaries in this text model.
        """
        sent1 = s.replace("?",".")
        sent2 = sent1.replace("!",".")
        sent3 = sent2.replace(",","")
        sent4 = sent3.replace(";","")
        sent5= sent4.replace(":", "")
        sent5 = sent5.replace("'", "")
        sentence = sent5.split(". ")
        for x in sentence:
            sent = x.split(" ")
            if len(sent) not in self.sentence_lengths:
                self.sentence_lengths[len(sent)] = 1
            else:
                self.sentence_lengths[len(sent)] += 1
        for y in s:
            if y in ".?!":
                if y not in self.punctuation:
                    self.punctuation[y] = 1
                else:
                    self.punctuation[y] += 1
            
            
        word_list = clean_text(s)
        for w in word_list:
            if w not in self.words:
                self.words[w] = 1
            else:
                self.words[w] += 1
        for r in word_list:
            if len(r) not in self.word_lengths:
                self.word_lengths[len(r)] = 1
            else:
                self.word_lengths[len(r)] += 1
         
        for l in word_list:
            x = stem(l)
            if x not in self.stems:
                self.stems[x] = 1
            else:
                self.stems[x] += 1
                                  
    def add_file(self,filename):
        """ adds all of the text in the file identified by filename to the model"""
        f = open(filename, 'r', encoding='utf8', errors='ignore')
        for line in f:
            line = line[:-1]
            self.add_string(line)
        f.close()
    def save_model(self):
         """saves the TextModel object self by writing its various feature dictionaries to files."""
         d = self.words
         f = open(self.name + "_" + "words", "w")
         f.write(str(d))
         f.close()
         
         e = self.word_lengths
         g = open(self.name + "_" + "word_lengths", "w")
         g.write(str(e))
         g.close()
         
         h = self.sentence_lengths
         i = open(self.name + "_" + "sentence_lengths", "w")
         i.write(str(h))
         i.close()
         
         j = self.stems
         k = open(self.name + "_" + "stems", "w")
         k.write(str(j))
         k.close()
         
         l = self.punctuation
         m = open(self.name + "_" + "punctuations", "w")
         m.write(str(l))
         m.close()
               
    def read_model(self):
        """eads the stored dictionaries for the called 
        TextModel object from their files and assigns them to
        the attributes of the called TextModel"""
        f = open(self.name + "_" + "words", "r")
        d_str = f.read()
        f.close()
        self.words = dict(eval(d_str))
        
        g = open(self.name + "_" + "word_lengths", "r")
        e_str = g.read()
        g.close()
        self.word_lengths = dict(eval(e_str))
        
        h = open(self.name + "_" + "sentence_lengths", "r")
        i_str = h.read()
        h.close()
        self.sentence_lengths = dict(eval(i_str))
        
        j = open(self.name + "_" + "stems", "r")
        k_str = j.read()
        j.close()
        self.stems = dict(eval(k_str))
        
        l = open(self.punctuation + "_" + "punctuations", "r")
        m_str = l.read()
        l.close()
        self.punctuation = dict(eval(m_str))
        

    def similarity_scores(self, other):
        """computes and returns a list of log
        similarity scores measuring the similarity of self and other"""
        word_score = compare_dictionaries(other.words, self.words)
        word_length_score = compare_dictionaries(other.word_lengths,self.word_lengths)
        sentence_lengths = compare_dictionaries(other.sentence_lengths,self.sentence_lengths)
        stems = compare_dictionaries(other.stems,self.stems)
        punctuation = compare_dictionaries(other.punctuation,self.punctuation)
        lst = [word_score] + [word_length_score] +[stems] + [sentence_lengths]  + [punctuation]
        return lst
    def classify(self, source1, source2):
        """ compares the called TextModel object 
        (self) to two other “source” TextModel objects
        (source1 and source2) and determines which of these other
        TextModels is the more likely source of the called TextModel."""
        scores1 = self.similarity_scores(source1)
        scores2 = self.similarity_scores(source2)
        print("scores for " + source1.name + ": " + str(scores1))
        print("scores for " + source2.name + ": " + str(scores2))
        count1 = 0
        count2 = 0
        for x in range(len(scores1)):
            if scores1[x] > scores2[x]:
                count1 += 1
            else:
                count2 += 1
        if count1>count2:
            print(self.name + " is more likely to have come from " + source1.name)
        else:
            print(self.name + " is more likely to have come from " + source2.name)
        
def run_tests():
    """ compares the documents and checks whether it works"""
    source1 = TextModel('friends_ep111')
    source1.add_file('friends_ep111.txt')
    source2 = TextModel('himym_ep1')
    source2.add_file('himym_ep1.txt')
    new1 = TextModel('friends_ep105')
    new1.add_file('friends_ep105.txt')
    new1.classify(source1, source2)
    
    source3 = TextModel('Shakespeare_JuliusCaesar')
    source3.add_file('Shakespeare_JuliusCaesar.txt')
    source4 = TextModel('JKrowling_Harry potter_1')
    source4.add_file('JKrowling_Harry potter_1.txt')
    new2 = TextModel('Shakespeare_Macbeth')
    new2.add_file('Shakespeare_Macbeth.txt')
    new2.classify(source3, source4)
    
    source5 = TextModel('times of india_article1')
    source5.add_file('times of india_article1.txt')
    source6 = TextModel('NDTV_article1')
    source6.add_file('NDTV_article1.txt')
    new7 = TextModel('times of india_article2')
    new7.add_file('times of india_article2.txt')
    new7.classify(source5, source6)
    
    source7 = TextModel('Final_ argument driven essay')
    source7.add_file('Final_ argument driven essay.txt')
    source8 = TextModel('Comparitive essay_sample')
    source8.add_file('Comparitive essay_sample.txt')
    new8 = TextModel('Comparitive analysis final draft')
    new8.add_file('Comparitive analysis final draft.txt')
    new8.classify(source7, source8)
    