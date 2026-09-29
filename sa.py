# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import pandas as pd
import pickle as pkl
import warnings
import seaborn as sns
from scipy.special import logit
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

class simulatedAnnealing:
    """
    Class for sampling from the Boltzmann distribution with decreasing temperature
    """
    def __init__(self, model, extraFeatures, initTemp, iterations, frequency, randomSeed=137, maximize=True):
        self.T0 = initTemp
        self.iterations = iterations
        self.randomSeed = randomSeed
        self.frequency = frequency
        self.maximize = maximize
        self.model = model
        self.extraFeatures = extraFeatures
        np.random.seed(seed=self.randomSeed)
            
    def calculateEnergy(self, modelFunc, input1, input2):
        if self.maximize == False:
            output1 = modelFunc(sequence = input1, mlModel = self.model, nonEmbed = self.extraFeatures)
            output2 = modelFunc(sequence = input2, mlModel = self.model, nonEmbed = self.extraFeatures)
            energyValue = output2-output1
            return(energyValue, output2, output1)
        else: 
            output1 = -modelFunc(sequence = input1, mlModel = self.model, nonEmbed = self.extraFeatures)
            output2 = -modelFunc(sequence = input2, mlModel = self.model, nonEmbed = self.extraFeatures)
#             print(output1, output2)
            energyValue = output2-output1
            return(energyValue, -output2, -output1)
    
#     def boltz(self, x):
#         return(np.exp(-1.0*x/self.T0))
    
    def proposeState(self, inputSeq, allowedPosIndices, printing = False):
#     testSeq = 'ACGTTCGACGGATTCACCGG'*4
        currentSeq = [letter for letter in inputSeq]
        proposedSeq = currentSeq.copy()
    #     allowedPosIndices = [i for i in range(len(inputSeq))]
        choosePosIndex = np.random.choice(allowedPosIndices)
        if printing == True:
            print("Current Sequence\n", inputSeq)
            print("Mutated Position Index:", choosePosIndex)
            print(currentSeq[choosePosIndex])
            allowedMut = ['A','C','G','T']
            allowedMut.remove(currentSeq[choosePosIndex])
            mutatedNuc = np.random.choice(allowedMut)
            print("Nucleotide after mutation:", mutatedNuc)
            proposedSeq[choosePosIndex] = mutatedNuc
        else:
            allowedMut = ['A','C','G','T']
            allowedMut.remove(currentSeq[choosePosIndex])
            mutatedNuc = np.random.choice(allowedMut)
            proposedSeq[choosePosIndex] = mutatedNuc
        return("".join(proposedSeq))

    def acceptState(self, currentSequence, modelFunc, allowedPosIndices, temp, txtFile, writeFile, printing = False):
        if printing == True:      
            print("Sequence Before Proposal:    ", currentSequence, file=writeFile)
            proposedSequence = self.proposeState(inputSeq=currentSequence, allowedPosIndices=allowedPosIndices, printing=printing)
            print("Sequence After Proposal:     ", proposedSequence, file=writeFile)
            energyProposed = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[0]
            probProposed = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[1]
            probInit = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[2]
            randomNumber = np.random.rand()
            boltzmann = np.exp(-1.0*energyProposed/temp)
            acceptBool = boltzmann > randomNumber
        #     The next commented lines are sanity checks
            print("Temperature", temp, file=writeFile)
            print("Boltzmann", np.exp(-energyProposed/temp), file=writeFile)
            print("Random Number", randomNumber, file=writeFile)
            print("Accpeted?", acceptBool, file=writeFile)
#             print(np.exp(-energyProposed/temp), randomNumber, acceptBool)
            if acceptBool:
                print("Sequence if accepted (or not)", proposedSequence, file=writeFile)   # to check
                return(proposedSequence, probProposed, probInit, boltzmann, int(acceptBool), randomNumber)
            else:
                print("Sequence if accepted (or not)", currentSequence, file=writeFile)   # to check
                return(currentSequence, probProposed, probInit, boltzmann, int(acceptBool), randomNumber)
        else: 
#             print("Sequence Current:", currentSequence)
            proposedSequence = self.proposeState(inputSeq=currentSequence, allowedPosIndices=allowedPosIndices, printing=printing)
#             print("Proposed Current:", proposedSequence)
            energyProposed = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[0]
            probProposed = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[1]
            probInit = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[2]
            randomNumber = np.random.rand()
            boltzmann = np.exp(-1.0*energyProposed/temp)
            acceptBool = boltzmann > randomNumber
        #     The next commented lines are sanity checks
            if acceptBool:
                return(proposedSequence, probProposed, probInit, boltzmann, int(acceptBool), randomNumber)
            else:
                return(currentSequence, probProposed, probInit, boltzmann, int(acceptBool), randomNumber)

    
    def markovChain(self, initialization, modelFunc, allowedPosIndices, txtFile, printing=False):
        sequenceList = []
        predictionList = np.zeros(self.iterations+1)
        nthSeq = initialization
        temperatures = np.zeros(self.iterations+1)
        Boltz = np.zeros(self.iterations+1)
        acceptance_bool = np.zeros(self.iterations+1)
        acceptance_rand = np.zeros(self.iterations+1)
        if printing == True:
            writing = open(txtFile, 'w')
            for i in range(1,self.iterations):
                T = self.T0/np.log(i+1)
                print("-"*100, file=writing)
                print("Iteration:", i, file=writing)
                nthSeq, modelPredProp, modelPredInit, boltz, acceptB, acceptR = self.acceptState(currentSequence = nthSeq, allowedPosIndices=allowedPosIndices, modelFunc=modelFunc, temp = T, txtFile=txtFile, writeFile=writing, printing=printing)
                print("Current Seq:                 ", nthSeq, file=writing)
                print("Proposed Score", modelPredProp, file=writing)
                print("Current Score ", modelPredInit, file=writing)
#                 print()
                acceptance_bool[i] = acceptB
                acceptance_rand[i] = acceptR
                predictionList[i] = modelPredInit
                Boltz[i] = boltz
                temperatures[i] = T
                if i % self.frequency == 0:
                    sequenceList.append(nthSeq)
        else:
            for i in range(1,self.iterations):
                T = self.T0/np.log(i+1)
                nthSeq, modelPred, modelPredInit, boltz, acceptB, acceptR = self.acceptState(currentSequence = nthSeq, allowedPosIndices=allowedPosIndices, modelFunc=modelFunc, temp = T, txtFile=txtFile, writeFile=writing)
                acceptance_bool[i] = acceptB
                acceptance_rand[i] = acceptR
                predictionList[i] = modelPredInit
                Boltz[i] = boltz
                temperatures[i] = T
                if i % self.frequency == 0:
                    sequenceList.append(nthSeq)
        return(initialization, modelFunc(sequence = initialization, mlModel = self.model, nonEmbed = self.extraFeatures), np.array(sequenceList), predictionList, temperatures,  acceptance_bool, Boltz, acceptance_rand)
    
class simulatedAnnealingPiecewise:
    """
    Class for sampling from the Boltzmann distribution with piecewise cooling rate
    """
    def __init__(self, model, initTemp, iterations, frequency, randomSeed=137, maximize=True):
        self.T0 = initTemp
        self.iterations = iterations
        self.randomSeed = randomSeed
        self.frequency = frequency
        self.maximize = maximize
        self.model = model
        np.random.seed(seed=self.randomSeed)
            
    def calculateEnergy(self, modelFunc, input1, input2):
        if self.maximize == False:
            output1 = modelFunc(sequence = input1, mlModel = self.model, nonEmbed = self.extraFeatures)
            output2 = modelFunc(sequence = input2, mlModel = self.model, nonEmbed = self.extraFeatures)
            energyValue = output2-output1
            return(energyValue, output2, output1)
        else: 
            output1 = -modelFunc(sequence = input1, mlModel = self.model, nonEmbed = self.extraFeatures)
            output2 = -modelFunc(sequence = input2, mlModel = self.model, nonEmbed = self.extraFeatures)
#             print(output1, output2)
            energyValue = output2-output1
            return(energyValue, -output2, -output1)
    
#     def boltz(self, x):
#         return(np.exp(-1.0*x/self.T0))
    
    def proposeState(self, inputSeq, allowedPosIndices, printing = False):
#     testSeq = 'ACGTTCGACGGATTCACCGG'*4
        currentSeq = [letter for letter in inputSeq]
        proposedSeq = currentSeq.copy()
    #     allowedPosIndices = [i for i in range(len(inputSeq))]
        choosePosIndex = np.random.choice(allowedPosIndices)
        if printing == True:
#             print("Current Sequence\n", currentSeq)
#             print("Mutated Position Index:", choosePosIndex)
#             print(currentSeq[choosePosIndex])
            allowedMut = ['A','C','G','T']
            allowedMut.remove(currentSeq[choosePosIndex])
            mutatedNuc = np.random.choice(allowedMut)
#             print("Nucleotide after mutation:", mutatedNuc)
            proposedSeq[choosePosIndex] = mutatedNuc
        else:
            allowedMut = ['A','C','G','T']
            allowedMut.remove(currentSeq[choosePosIndex])
            mutatedNuc = np.random.choice(allowedMut)
            proposedSeq[choosePosIndex] = mutatedNuc
        return("".join(proposedSeq))

    def acceptState(self, currentSequence, modelFunc, allowedPosIndices, temp, txtFile, writeFile, printing = False):
        if printing == True:      
            print("Sequence Before Proposal:    ", currentSequence, file=writeFile)
            proposedSequence = self.proposeState(inputSeq=currentSequence, allowedPosIndices=allowedPosIndices, printing=printing)
            print("Sequence After Proposal:     ", proposedSequence, file=writeFile)
            energyProposed = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[0]
            probProposed = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[1]
            probInit = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[2]
            randomNumber = np.random.rand()
            boltzmann = np.exp(-1.0*energyProposed/temp)
            acceptBool = boltzmann > randomNumber
        #     The next commented lines are sanity checks
            print("Temperature", temp, file=writeFile)
            print("Boltzmann", np.exp(-energyProposed/temp), file=writeFile)
            print("Random Number", randomNumber, file=writeFile)
            print("Accpeted?", acceptBool, file=writeFile)
#             print(np.exp(-energyProposed/temp), randomNumber, acceptBool)
            if acceptBool:
                print("Sequence if accepted (or not)", proposedSequence, file=writeFile)   # to check
                return(proposedSequence, probProposed, probInit, boltzmann, int(acceptBool), randomNumber)
            else:
                print("Sequence if accepted (or not)", currentSequence, file=writeFile)   # to check
                return(currentSequence, probProposed, probInit, boltzmann, int(acceptBool), randomNumber)
        else: 
#             print("Sequence Current:", currentSequence)
            proposedSequence = self.proposeState(inputSeq=currentSequence, allowedPosIndices=allowedPosIndices, printing=printing)
#             print("Proposed Current:", proposedSequence)
            energyProposed = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[0]
            probProposed = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[1]
            probInit = self.calculateEnergy(modelFunc=modelFunc, input1 = currentSequence, input2 = proposedSequence)[2]
            randomNumber = np.random.rand()
            boltzmann = np.exp(-1.0*energyProposed/temp)
            acceptBool = boltzmann > randomNumber
        #     The next commented lines are sanity checks
            if acceptBool:
                return(proposedSequence, probProposed, probInit, boltzmann, int(acceptBool), randomNumber)
            else:
                return(currentSequence, probProposed, probInit, boltzmann, int(acceptBool), randomNumber)

    
    def markovChain(self, initialization, modelFunc, allowedPosIndices, txtFile, printing=False):
        sequenceList = []
        predictionList = np.zeros(self.iterations+1)
        nthSeq = initialization
        temperatures = np.zeros(self.iterations+1)
        Boltz = np.zeros(self.iterations+1)
        acceptance_bool = np.zeros(self.iterations+1)
        acceptance_rand = np.zeros(self.iterations+1)
        initScore = modelFunc(sequence = initialization, mlModel = self.model, nonEmbed = self.extraFeatures)
        if printing == True:
            writing = open(txtFile, 'w')
            pastThresh = False
            for i in range(1,self.iterations):
                nthScore = modelFunc(sequence = nthSeq, mlModel = self.model)
                if (i < int(self.iterations/4) or nthScore < 1.1*initScore) and pastThresh == False:
                    T = self.T0/np.log(i+1)
                    print("-"*100, file=writing)
                    print("Iteration:", i, file=writing)
                    nthSeq, modelPredProp, modelPredInit, boltz, acceptB, acceptR = self.acceptState(currentSequence = nthSeq, allowedPosIndices=allowedPosIndices, modelFunc=modelFunc, temp = T, txtFile=txtFile, writeFile=writing, printing=printing)
                    print("Current Seq:                 ", nthSeq, file=writing)
                    print("Proposed Score", modelPredProp, file=writing)
                    print("Current Score ", modelPredInit, file=writing)
    #                 print()
                    acceptance_bool[i] = acceptB
                    acceptance_rand[i] = acceptR
                    predictionList[i] = modelPredInit
                    Boltz[i] = boltz
                    temperatures[i] = T
                    if i % self.frequency == 0:
                        sequenceList.append(nthSeq)

                elif (i >= int(self.iterations/4) and nthScore >= 1.1*initScore) and pastThresh == False:
                    T = self.T0/np.log(i+1)
                    print("-"*100, file=writing)
                    print("Iteration:", i, file=writing)
                    nthSeq, modelPredProp, modelPredInit, boltz, acceptB, acceptR = self.acceptState(currentSequence = nthSeq, allowedPosIndices=allowedPosIndices, modelFunc=modelFunc, temp = T, txtFile=txtFile, writeFile=writing, printing=printing)
                    print("Current Seq:                 ", nthSeq, file=writing)
                    print("Proposed Score", modelPredProp, file=writing)
                    print("Current Score ", modelPredInit, file=writing)
    #                 print()
                    acceptance_bool[i] = acceptB
                    acceptance_rand[i] = acceptR
                    predictionList[i] = modelPredInit
                    Boltz[i] = boltz
                    temperatures[i] = T
                    if i % self.frequency == 0:
                        sequenceList.append(nthSeq)
                    iSpecial = i
                    pastThresh = True
                    
                elif pastThresh == True:
                    slope = -self.T0/(iSpecial*np.log(iSpecial)*np.log(iSpecial))
                    intercept = self.T0/np.log(iSpecial)
                    T = slope*(i-iSpecial)+intercept
                    print("-"*100, file=writing)
                    print("Iteration:", i, file=writing)
                    nthSeq, modelPredProp, modelPredInit, boltz, acceptB, acceptR = self.acceptState(currentSequence = nthSeq, allowedPosIndices=allowedPosIndices, modelFunc=modelFunc, temp = T, txtFile=txtFile, writeFile=writing, printing=printing)
                    print("Current Seq:                 ", nthSeq, file=writing)
                    print("Proposed Score", modelPredProp, file=writing)
                    print("Current Score ", modelPredInit, file=writing)
    #                 print()
                    acceptance_bool[i] = acceptB
                    acceptance_rand[i] = acceptR
                    predictionList[i] = modelPredInit
                    Boltz[i] = boltz
                    temperatures[i] = T
                    if i % self.frequency == 0:
                        sequenceList.append(nthSeq)
                    
        else:
            for i in range(1,self.iterations):
                T = self.T0/np.log(i+1)
                nthSeq, modelPred, modelPredInit, boltz, acceptB, acceptR = self.acceptState(currentSequence = nthSeq, allowedPosIndices=allowedPosIndices, modelFunc=modelFunc, temp = T, txtFile=txtFile, writeFile=writing)
                acceptance_bool[i] = acceptB
                acceptance_rand[i] = acceptR
                predictionList[i] = modelPredInit
                Boltz[i] = boltz
                temperatures[i] = T
                if i % self.frequency == 0:
                    sequenceList.append(nthSeq)
        return(initialization, modelFunc(sequence = initialization, mlModel = self.model), np.array(sequenceList), predictionList, temperatures,  acceptance_bool, Boltz, acceptance_rand)
