#!/usr/bin/env python
# coding: utf-8

# Authors: Harsha Raja Shivakumar (hrajash) | Maithreyi Prabhu (mmanurn) | Sunny Bhati (sbhati)

# Spam Classification using Naive Bayes Algorithm

import re
import os
from collections import Counter
import numpy as np
import sys



def pre_processing(text):
    
    # Convert string to lower case
    text1 = str(text).lower()
    
    # Remove the content till subject
    try:
        text1 = text1[text1.index("subject"):len(text1)]
    except:
        text1 = text1
    
    # Replacing \n
    text1 = str(text1).replace("\n", " ")
    
    # Replacing \t
    text1 = str(text1).replace("\t", " ")
    
    # Removing html tags and content inside it
    text1 = re.sub(r'<.*?>', '', text1)
    
    # Removing URLs
    text1 = re.sub(r'[a-z]*[:.]+\S+', '', text1)
    
    # Remove punctuations
    text1 = re.sub(r'[^\w\s'']',' ',text1)
    
    # Remove hyphen
    text1 = re.sub('_', '',text1)
    
    # Removing digits
    text1 = re.sub("\d+", "", text1)
    
    # Removing words with 2 letters and less
    text1 = re.sub(r'\b\w{1,2}\b', '',text1)
    
    text1 = re.sub('nbsp', '',text1)
    
    return text1



def read_files(directory):
    data = {}
    for files in os.listdir(directory):
        with open(os.getcwd() + "/" + directory + "/" + files, "r", encoding="Latin-1") as f:
            #data[files] = pre_processing(f.read()) 
            data[files] = f.read()
    return data



def words(doc1):
    """
    Function to Split the document to words
    
    Parameters:
    doc1 - documents to be split
    
    Returns:
    List of words for each document
    
    """
    return[word for sent in doc1 for word in sent.split()]



def word_count_maxl(words):
    """
    Function to calculate maximum likelihood of a word given that class
    
    Parameters:
    words_count - word counts of a class
    words_all   - All the words in that class
    
    Returns:
    Maximum likelihood of a word given the class
    
    """
    words_mle = dict(Counter(words))
    
    # Removing sparse words
    #words_mle = dict(filter(lambda elem: elem[1] > 10, words_mle.items()))
    
    total_words = sum(words_mle.values())
    
    words_mle_final = {}
    for words in list(words_mle.keys()):
        words_mle_final[words] = np.log(words_mle[words]/total_words)
        
    
    return words_mle_final

if __name__== "__main__":
    if(len(sys.argv) != 4):
        raise Exception("usage: ./spam.py training-directory testing-directory output-file")

    train_spam = read_files(sys.argv[1] + '/spam')
            
    train_notspam = read_files(sys.argv[1] + '/notspam')

    test = read_files(sys.argv[2])

    spam_maxl = word_count_maxl(words(list(train_spam.values())))
    notspam_maxl = word_count_maxl(words(list(train_notspam.values())))



    final_class = {}
    prob_spam = np.log(len(train_spam)/(len(train_spam) + len(train_notspam)))
    prob_notspam = np.log(len(train_notspam)/(len(train_spam) + len(train_notspam)))


    for key,sent in test.items():
        words = sent.split()
        class_spam = 0
        class_notspam = 0
        class_spam += prob_spam
        class_notspam += prob_notspam
        
        for word in words:
            if word in spam_maxl:
                class_spam += spam_maxl[word]
            else:
                class_spam += np.log(pow(10, -20))


            if word in notspam_maxl:
                class_notspam += notspam_maxl[word]
            else:
                class_notspam += np.log(pow(10, -20))
        
        if class_notspam > class_spam:
            final_class[key] = 'notspam'
        elif class_notspam == class_spam:
            temp = random.randint(0, 1)
            if temp == 1:
                final_class[key] = 'spam'
            else:
                final_class[key] = 'notspam'
        else:
            final_class[key] = 'spam'


    actual_test_class = {}
    with open('test-groundtruth.txt', "r", encoding="Latin-1") as f:
        sentences = f.read()
        for each in sentences.split("\n"):
            split = each.split(" ")
            if len(split) > 1:
                actual_test_class[split[0]] = split[1]


    # Accuracy
    print("Accuracy : ",len(set(final_class.items()).intersection(actual_test_class.items()))/len(actual_test_class)*100)

    # Writing to output file
    with open(sys.argv[3], "w") as file:
        for key, value in final_class.items():
            print(key, value, file=file)


