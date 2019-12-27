###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np
from collections import defaultdict
from collections import Counter
import pandas as pd


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    initial_state_distribution=[0]*12
    final_state_distribution=[0]*12
    emission_count={}
    type_of_words=['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
    transition_count=[[0 ]*12 for i in range(12)]
    transition_probability=[[0]*12 for i in range(12)]
    emission_probability={}
    count=float(1/100000000000000000)
    count1=0
    kkl=500
    trans_s1_sn=[[0]*12 for i in range(12)]
    trans_sn_1_sn=[[0]*12 for i in range(12)]
    list_of_mcmc=[]

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            p=self.simplified(sentence)[1]
            return(sum(p))
        elif model == "Complex":
            return ((self.hmm_viterbi(sentence)[1]+sum(self.simplified(sentence)[1]))/2)
        elif model == "HMM":
            return(self.hmm_viterbi(sentence)[1])
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, f):
        initial_count=[0]*12
        first_word_count=[0]*12
        last_word_count=[0]*12
        self.total_num_words=0
        total_num_sentances=0
        self.prior_count=[]
        dict_1={}
        self.emission_count=self.emission_probability={j:[Solver.count]*12 for i in f for j in i[0]}
        for i in range(len(f)):
            self.total_num_words+=len(f[i][0])
            total_num_sentances+=1
            #Counts used for calculating initial state distribution
            first_word_count[Solver.type_of_words.index(f[i][1][0])]+=1
            last_word_count[Solver.type_of_words.index(f[i][1][-1])]+=1
            for j in range(len(f[i][1])):
                index=Solver.type_of_words.index(f[i][1][j])
                initial_count[index]+=1
                #Counts used for calculating transition probability
                if(j<len(f[i][1])-1):
                    Solver.transition_count[index][Solver.type_of_words.index(f[i][1][j+1])]+=1

                #Counts used for calculating emission probability
                self.emission_count[f[i][0][j]][index]+=1

        for i in range(12):
            #Initial Probability distribution
            # self.prior_count.append(sum(Solver.transition_count[i]))

            if(first_word_count[i]==0):
                Solver.initial_state_distribution[i]=Solver.count
            else:
                Solver.initial_state_distribution[i]=float(first_word_count[i])/len(f)

            if(last_word_count[i]==0):
                Solver.final_state_distribution[i]=Solver.count
            else:
                Solver.final_state_distribution[i]=float(last_word_count[i]/len(f))
            #Calculating transition probabilities
            for j in range(12):
                if(sum(Solver.transition_count[i])==0 or Solver.transition_count[i][j]==0):
                    Solver.transition_probability[i][j]=Solver.count
                else:
                    Solver.transition_probability[i][j]=float(Solver.transition_count[i][j])/sum(Solver.transition_count[i])

        self.prior_probability=[]
        dfg=0

        for i in range(12):
            self.prior_probability.append(float(initial_count[i])/self.total_num_words)

        for i in self.emission_probability:
            for j in range(12):
                if(initial_count[j]==0 or self.emission_count[i][j]==0):
                    self.emission_probability[i][j]=Solver.count
                else:
                    self.emission_probability[i][j]=float(self.emission_count[i][j])/initial_count[j]



        dict_1={j:[Solver.count1]*12 for j in Solver.type_of_words}
        dict_2={j:[Solver.count1]*12 for j in Solver.type_of_words}

        for i in range(len(Solver.type_of_words)):
            dict_1={j:[Solver.count1]*12 for j in Solver.type_of_words}
            for k in range(len(f)):
                if f[k][1][len(f[k][0])-1]==Solver.type_of_words[i]:
                    dict_1[f[k][1][0]][Solver.type_of_words.index(f[k][1][len(f[k][0])-2])]+=1
            Solver.list_of_mcmc.append(dict_1)
      


        for k in range(len(f)):
            dict_2[f[k][1][0]][Solver.type_of_words.index(f[k][1][len(f[k][0])-2])]+=1
    



        for i in range(0,len(Solver.list_of_mcmc)):
            for j in Solver.type_of_words:
                for k in range(0,12):
                    if Solver.list_of_mcmc[i][j][k]==0:
                        Solver.list_of_mcmc[i][j][k]=float(Solver.count)
                    else:
                        Solver.list_of_mcmc[i][j][k]=float(Solver.list_of_mcmc[i][j][k]/dict_2[j][k])



        for i in range(len(f)):
            Solver.trans_s1_sn[Solver.type_of_words.index(f[i][1][0])][Solver.type_of_words.index(f[i][1][len(f[i][0])-1])]+=1
            Solver.trans_sn_1_sn[Solver.type_of_words.index(f[i][1][len(f[i][0])-2])][Solver.type_of_words.index(f[i][1][len(f[i][0])-1])]+=1


        for i in range(12):

            for j in range(12):
                if(sum(Solver.trans_s1_sn[i])==0 or Solver.trans_s1_sn[i][j]==0):
                    Solver.trans_s1_sn[i][j]=Solver.count
                else:
                    Solver.trans_s1_sn[i][j]=float(Solver.trans_s1_sn[i][j])/sum(Solver.trans_s1_sn[i])
                if(sum(Solver.trans_sn_1_sn[i])==0 or Solver.trans_sn_1_sn[i][j]==0):
                    Solver.trans_sn_1_sn[i][j]=Solver.count
                else:
                    Solver.trans_sn_1_sn[i][j]=float(Solver.trans_sn_1_sn[i][j])/sum(Solver.trans_sn_1_sn[i])



    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        part_of_speech=[]
        posterior=[]
        for i in range(len(sentence)):
            self.most_probable_tag=[0]*12
            if(sentence[i] not in self.emission_probability):
                #If word not seen before, provide high probability that, that word is a noun.
                self.emission_probability[sentence[i]]=[Solver.count,Solver.count,Solver.count,Solver.count,Solver.count,1-(Solver.count)*11,Solver.count,Solver.count,Solver.count,Solver.count,Solver.count,Solver.count]
            for j in range(12):
                self.most_probable_tag[j]= math.log(self.emission_probability[sentence[i]][j])+math.log(self.prior_probability[j])
                # Solver.initial_state_distribution[j]

            part_of_speech.append(Solver.type_of_words[self.most_probable_tag.index(max(self.most_probable_tag))])
            posterior.append(max(self.most_probable_tag))

        return  (part_of_speech,posterior)

    def mostCommon(lst):
        return [Counter(col).most_common(1)[0][0] for col in zip(*lst)]


    def complex_mcmc(self, sentence):
        T=len(sentence)
        initial_sample=["noun"]*T
        top_samples=[]
        p1=[]
        p2=[]
        df=[]




        for no_of_iteration in range(Solver.kkl):
            for i in range(0,T):
                p1=[]
                probability_values=[]
                probablility_sum=0

                for j in Solver.type_of_words:

                    if len(sentence)>2:


                        if i==0:
                            ak1 = Solver.initial_state_distribution[Solver.type_of_words.index(j)]
                            b1=self.emission_probability[sentence[i]][Solver.type_of_words.index(j)]
                            c1=self.transition_probability[Solver.type_of_words.index(j)][Solver.type_of_words.index(initial_sample[i+1])]
                            d1=self.list_of_mcmc[Solver.type_of_words.index(initial_sample[T-1])][j][Solver.type_of_words.index(initial_sample[T-2])]
                            p=float(ak1*b1*c1*d1)

                        if i >0 and i <T-2:
                            e1=self.emission_probability[sentence[i]][Solver.type_of_words.index(j)]
                            f1=self.transition_probability[Solver.type_of_words.index(j)][Solver.type_of_words.index(initial_sample[i+1])]
                            g1=self.transition_probability[Solver.type_of_words.index(initial_sample[i-1])][Solver.type_of_words.index(j)]
                            p=float(e1*f1*g1)
                        if i==T-1:
                            h1=self.emission_probability[sentence[i]][Solver.type_of_words.index(j)]
                            k1=self.list_of_mcmc[Solver.type_of_words.index(j)][initial_sample[0]][Solver.type_of_words.index(initial_sample[T-2])]
                            p=float(h1*k1)
                        if i== T-2:
                            l1=self.emission_probability[sentence[i]][Solver.type_of_words.index(j)]
                            q1=self.list_of_mcmc[Solver.type_of_words.index(initial_sample[T-1])][initial_sample[0]][Solver.type_of_words.index(j)]
                            z1=self.transition_probability[Solver.type_of_words.index(initial_sample[T-3])][Solver.type_of_words.index(j)]
                            p=float(l1*q1*z1)

                    if len(sentence)==1:
                        a1 = Solver.initial_state_distribution[Solver.type_of_words.index(j)]
                        b1=self.emission_probability[sentence[i]][Solver.type_of_words.index(j)]
                        p=float(a1*b1)
                    if len(sentence)==2:
                        if i==0:
                            ak1 = Solver.initial_state_distribution[Solver.type_of_words.index(j)]
                            b1=self.emission_probability[sentence[i]][Solver.type_of_words.index(j)]
                            c1=self.transition_probability[Solver.type_of_words.index(j)][Solver.type_of_words.index(initial_sample[i+1])]
                            p=float(ak1*b1*c1)
                        if i==1:
                            g1=self.transition_probability[Solver.type_of_words.index(initial_sample[i-1])][Solver.type_of_words.index(j)]
                            e1=self.emission_probability[sentence[i]][Solver.type_of_words.index(j)]
                            p=float(g1*e1)




                    probablility_sum += p
                    probability_values.append(p)


                c=0
                r = random.uniform(0.00,1.00)
    

                for q in range(0, len(probability_values)):
                    probability_values[q] =(probability_values[q]/probablility_sum)
                    c += probability_values[q]
                    probability_values[q] = c
                    if r < probability_values[q]:
                        o = q
                        break
                initial_sample[i]=Solver.type_of_words[o]


            top_samples.append(initial_sample)

        del top_samples[:100]

        count_list=[]

        df= Solver.mostCommon(top_samples)

        for i in range(0,T):
            fghi=0
            for k in range(0,len(top_samples)):
                if top_samples[k][i]==df[i]:
                    fghi=fghi+1
            count_list.append(fghi)




        return(df)








    def hmm_viterbi(self, sentence):
        T=len(sentence)

        result = [[0 for t in range(T)]for j in range(12)]

        memo = [[0 for i in range(12)]for t in range(T)]

        for t in range(T):
            for j in range(12):
                if t==0:
                    memo[t][j] = math.log(Solver.initial_state_distribution[j]) + math.log(self.emission_probability[sentence[t]][j])
                else:
                    cost = [memo[t - 1][i]+math.log(Solver.transition_probability[i][j] )for i in range(12)]
                    maxc = max(cost)
                    memo[t][j] = math.log(self.emission_probability[sentence[t]][j]) + maxc
                    result[j][t] = cost.index(maxc)
        string2 = []

        idx = memo[T - 1].index(max(memo[T - 1]))

        string2.append(Solver.type_of_words[idx])
        i = len(sentence) - 1

        while (i > 0):
            idx = result[idx][i]
            string2.append(Solver.type_of_words[idx])
            i -= 1
        return (string2[::-1],max(memo[T - 1]))


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)[0]
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)[0]
        else:
            print("Unknown algo!")
