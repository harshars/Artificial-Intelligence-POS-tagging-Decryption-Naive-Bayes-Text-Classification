#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: Harsha Raja Shivakumar (hrajash) | Maithreyi Prabhu (mmanurn) | Sunny Bhati (sbhati)
#
# based on skeleton code by D. Crandall, 11/2019
#
# ./break_code.py : attack encryption
#

import math
import copy 
import sys
import encode
from collections import Counter
import string
import numpy as np
import random
import itertools
import re
from copy import deepcopy

def find_probabilities(document,prob_letter_comb,prob_letter):  
    sum_document = 0
    for word in document.split():
        sum_of_word = prob_letter[word[0]]
        for i in range(1,len(word)):
            sum_of_word += prob_letter_comb[word[i-1]+word[i]]
        sum_document += sum_of_word
        
    return sum_document

def parse_encrypted(filename):
    with open(filename, "r") as f:
        return f.read()

def parse_text(filename):
    with open(filename, "r") as f:
        return([line for line in f.read().split("\n")])

def break_code(data_encrypted, data):

	letters = [i[:1] for i in data]
	letter_count = dict(Counter(letters))

	# Probability of a letter
	prob_letter = {}
	for key,value in letter_count.items():
	    prob_letter[key] = np.log(letter_count[key]/sum(letter_count.values()))

	prob_letter[' '] = np.log(pow(10,-8))

	data_combined = ' '.join([i for i in data])

	letters_unique = [chr(x) for x in range(ord('a'), ord('z') + 1)] 

	letters_unique.append(' ')

	letters_comb = {i+j:0 for i in letters_unique for j in letters_unique}

	# Transition Probabilities of letters
	for i in range(len(data_combined)-1):
		key = data_combined[i]+data_combined[i+1]
		letters_comb[key] = letters_comb[key]+1

	replace_table_temp = random.sample(range(97,123), 26)

	replace_table_temp.append(32)
    

	prob_letter_comb = {keys: np.log(values/sum(letters_comb.values())) if values!=0 else np.log(pow(10,-20)) \
						for keys, values in letters_comb.items()}

	replace_table = {letters_unique[i]: chr(replace_table_temp[i]) for i in range(len(letters_unique))}

	rearrange_table = [0, 1, 2, 3]

	data_encrypted = re.sub('\n','', data_encrypted)

	count = 0
	flag = 0
	best = ''


	decrypt_t = encode.encode(data_encrypted, replace_table, rearrange_table)
	prob_d = find_probabilities(decrypt_t, prob_letter_comb, prob_letter)



	while True: 
	    
	    count += 1
	    if count == 20000:
	    	print(encode.encode(data_encrypted, replace_table, rearrange_table))
	    	return(encode.encode(data_encrypted, replace_table, rearrange_table))
	   
	    if np.random.uniform() > 0.5:
	        flag = 1
	        rearrange_table_old = deepcopy(rearrange_table)
	        swap_indices = random.sample(range(4),2)
	        temp = rearrange_table[swap_indices[0]]
	        rearrange_table[swap_indices[0]] = rearrange_table[swap_indices[1]]
	        rearrange_table[swap_indices[1]] = temp
	    else:
	        flag = 0
	        replace_table_old = deepcopy(replace_table)
	        swap_indices = random.sample(range(97,123),2)
	        temp = replace_table[chr(swap_indices[0])]
	        replace_table[chr(swap_indices[0])] = replace_table[chr(swap_indices[1])]
	        replace_table[chr(swap_indices[1])] = temp
	        
	        
	    decrypt_t_prime = encode.encode(data_encrypted, replace_table, rearrange_table)
	    prob_d_prime = find_probabilities(decrypt_t_prime, prob_letter_comb, prob_letter) 
	    

	    if prob_d_prime > prob_d:
	        prob_d = prob_d_prime
	        best = decrypt_t_prime
	    else:
	        if np.random.binomial(1, np.exp(prob_d_prime-prob_d)) == 0:
	            if flag == 1:
	                rearrange_table = deepcopy(rearrange_table_old)
	            else:
	                replace_table = deepcopy(replace_table_old)
	        else:
	            prob_d = prob_d_prime




if __name__== "__main__":
    if(len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = parse_encrypted(sys.argv[1])
    corpus = parse_text(sys.argv[2])
    decoded = break_code(encoded, corpus)

    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)

