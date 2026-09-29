# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import warnings
import seaborn as sns
from pandas import DataFrame
from scipy.special import logit
import matplotlib.pyplot as plt
from multiprocessing import Pool
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

from sa import simulatedAnnealing

warnings.filterwarnings("ignore")

p = argparse.ArgumentParser(description='Parse the arguments')
p.add_argument('-r', '--rank', type=int, help='Rank of sequence that is used to start Markov chain', required=True)
p.add_argument('-o', '--outdir', type=str, help='Output directory to store the outputs', required=True)
p.add_argument('-g', '--gene', type=str, help='Gene of sequence to start the Markov chain', required=True)
p.add_argument('-p', '--prime', type=str, help='List of gene sequences corresponding to the priming regions (5 prime to 3 prime)', required=True)
args = p.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M", output_hidden_states=True, return_dict=True)
bertModel = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
bertTokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

def embed(sequence, poolType, tokenizer, model):
    '''
    get either 'mean' or 'max' embeddings for a given sequence, tokenizer and model
    '''
    inputs = tokenizer(sequence, return_tensors = 'pt')["input_ids"]
    outputs = model(inputs)
    hidden_states = outputs[0] 
    # hidden_states = outputs[2] 
    # layer = hidden_states[1]
    if poolType == 'mean':
        # return torch.mean(layer, dim=0).detach().numpy()
        return torch.mean(hidden_states[0], dim=0).detach().numpy()
    elif poolType == 'max':
        # return torch.max(layer, dim=0)[0].detach().numpy()
        return torch.max(hidden_states[0], dim=0)[0].detach().numpy()
    
df = pd.read_csv('pathToDataFile.csv')

bins = [-0.1, 99.99, 100.1] 
labels = [0, 100]

# Categorize the data
df['categorized_percentage'] = pd.cut(df['Amp'], bins=bins, labels=labels, right=False)
df['categorized_percentage'] = df['categorized_percentage'].astype(float)

# Create maxEmb DataFrame including 'primer' and 'ct'
maxEmb = pd.DataFrame(df['maxEmbed'].to_list())
maxEmb['categorized_percentage'] = df['categorized_percentage']
maxEmb['Gene'] = df['Gene']
maxEmb['Seq'] = df['Sequence']

# Perform downstream analysis on numerical (not one-hot encoded variables)
x = maxEmb.drop(['categorized_percentage', 'Gene','Seq'], axis=1).values 
y = maxEmb['categorized_percentage'].values

bertColumns = [f'BERT{i+1}' for i in range(x.shape[1])]

bertDF = pd.DataFrame(data=x, columns=bertColumns)
labeledBertDF = pd.concat([bertDF, pd.DataFrame(data=y, columns=['categorized_percentage'])], axis=1)

labeledBertDF['stratify_col'] = labeledBertDF['categorized_percentage'].astype(str) + '_' + df['Gene'].astype(str)                
labeledBertDF['Seq'] = maxEmb['Seq']                 
labeledBertDF['Gene'] = maxEmb['Gene']

random_state = 7722

# Split the data into training and testing sets (80% train, 20% test), keeping equal proportion of genes that amplify
X_train, X_temp, y_train, y_temp = train_test_split(labeledBertDF.drop(['categorized_percentage'], axis=1),
                                                    labeledBertDF['categorized_percentage'],
                                                    test_size=0.2,
                                                    random_state=random_state,
                                                    stratify=labeledBertDF['stratify_col'])

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=X_temp['stratify_col'])
X_train_bert = X_train[bertColumns]
X_val_bert = X_val[bertColumns]
X_test_bert = X_test[bertColumns]

scaledFit = StandardScaler().fit(X_train_bert)
X_train_bertz = scaledFit.transform(X_train_bert)
X_val_bertz = scaledFit.transform(X_val_bert)
X_test_bertz = scaledFit.transform(X_test_bert)

pca = PCA(n_components=30, random_state=random_state)
pcaFit = pca.fit(X_train_bertz)

X_train_pca = pcaFit.transform(X_train_bertz)
X_val_pca = pcaFit.transform(X_val_bertz)
X_test_pca = pcaFit.transform(X_test_bertz)

scaledFitPCA = StandardScaler().fit(X_train_pca)
X_train_pcaz = scaledFitPCA.transform(X_train_pca)
X_val_pcaz = scaledFitPCA.transform(X_val_pca)
X_test_pcaz = scaledFitPCA.transform(X_test_pca)

X_train_pca_df = pd.DataFrame(X_train_pcaz, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
X_val_pca_df = pd.DataFrame(X_val_pcaz, columns=[f'PC{i+1}' for i in range(X_val_pca.shape[1])])
X_test_pca_df = pd.DataFrame(X_test_pcaz, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])



X_train_df = pd.concat([X_train_pca_df, 
                    pd.DataFrame(data=y_train, columns=['categorized_percentage']).reset_index(drop=True),
                    X_train[['Seq','Gene']].reset_index(drop=True)],
                    axis=1)

X_val_df = pd.concat([X_val_pca_df,
                      pd.DataFrame(data=y_val, columns=['categorized_percentage']).reset_index(drop=True),
                      X_val[['Seq','Gene']].reset_index(drop=True)],
                      axis=1)

X_test_df = pd.concat([X_test_pca_df, 
                       pd.DataFrame(data=y_test, columns=['categorized_percentage']).reset_index(drop=True),
                       X_test[['Seq','Gene']].reset_index(drop=True)], 
                       axis=1)

# print(X_train_df.shape)

featureCols = [term for term in X_train_df.columns if ('categorized_percentage' not in term and 'Seq' not in term and 'Gene' not in term)]

train_data = X_train_df[featureCols]
train_data['target'] = X_train_df['categorized_percentage']

# Separate the majority and minority classes
majority_class = train_data[train_data['target'] == 100]  
minority_class = train_data[train_data['target'] == 0] 

# Oversample the minority class
minority_oversampled = resample(minority_class,
                                 replace=True,  
                                 n_samples=len(majority_class), 
                                 random_state=random_state) 



# Combine majority class with the oversampled minority class
train_oversampled = pd.concat([majority_class, minority_oversampled])

# Shuffle the resulting DataFrame
train_oversampled = train_oversampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

# Separate the features and target
X_train_df = train_oversampled.drop('target', axis=1)
y_train_df = train_oversampled['target'].values


X_val_df = X_val_df.drop('categorized_percentage', axis=1) 
X_test_df = X_test_df.drop('categorized_percentage', axis=1) 

X_train = X_train_df
y_train = y_train_df
X_val = X_val_df
X_test = X_test_df
fitModel = pkl.load(open('modelFiles/rf_bert_features.pkl','rb'))

def primeIndices(sequence, primeArray):
    indices = []
    for prime in primeArray:
        for i in range(len(sequence)-len(prime)+1):
            if sequence[i:i+len(prime)] == prime:
                indices+=[term+1 for term in range(i,i+len(prime))]
    return(indices)

def seqToModelOutput(sequence, mlModel, modelType = 'prob'):
    dbert2 = np.array([embed(sequence = sequence, poolType='max', tokenizer=bertTokenizer, model=bertModel)])
    dbert2z = scaledFit.transform(dbert2)
    dbert2pca = pcaFit.transform(dbert2z)
    if modelType == 'prob':
        modelOutput = logit(mlModel.predict_proba(dbert2pca)[:,1][0])
    else:
        modelOutput = logit(mlModel.predict(dbert2pca)[0])
    return(modelOutput)

primeList = args.prime.split(',')
gene = args.gene

predictions = fitModel.predict_proba(X_val[featureCols].values)[:,1]
observations = y_val
predDF = DataFrame(predictions, columns = ['Prob'])
predDF['Obs'] = y_val.values
predDF['Seq'] = X_test['Seq']
predDF['Gene'] = X_test['Gene']

topPredDF = predDF[(predDF['Prob'] > 0.70) & (predDF['Obs'] == 100) & (predDF['Gene'] == gene)].sort_values(by='Prob', ascending=False)

seqAbove75 = topPredDF['Seq'].values
predAbove75 = topPredDF['Prob'].values

# print(seqAbove75)

# t0 = float(args.temp)
seqIdx = int(args.rank)

indexGene = primeIndices(sequence=seqAbove75[seqIdx], primeArray=primeList)

print(indexGene)

def allMut(sequence, indices):
    allSingleMut = []
    currentSeq = [letter for letter in sequence]
    #     allowedPosIndices = [i for i in range(len(inputSeq))]
    for term in indices:
        allowedMut = ['A','C','G','T']
        allowedMut.remove(currentSeq[term])
        for letter in allowedMut:
            currentSeq[term] = letter
            allSingleMut.append(''.join(currentSeq))
    return(allSingleMut)

singleMutations = allMut(seqAbove75[seqIdx], indices = indexGene)

saDummy = simulatedAnnealing(model=fitModel, initTemp = 0.7, iterations = len(indexGene)*90, frequency = 1)
energies = []
for i in range(len(singleMutations)):
    eValue = saDummy.calculateEnergy(modelFunc = seqToModelOutput, input1 = seqAbove75[seqIdx], input2 = singleMutations[i])
    energies.append(eValue[0])

percent95 = sorted(energies)[int(0.95*len(energies))]
percent25 = max(0.01, sorted(energies)[int(0.25*len(energies))])
upperT = max(1.5*percent95, 1.25*max(energies))
tempRange = [np.round(term, decimals = 3) for term in np.linspace(percent25, upperT, 15)]

print(f'95 percentile = {percent95}, 25 percentile = {percent25}')
print('Temperature Range')
print(tempRange)

os.system(f'mkdir -p {args.outdir}')

def saTemp(temperature):
    SA = simulatedAnnealing(model=fitModel, initTemp = temperature, iterations = len(indexGene)*90, frequency = 1)
    saOutputs = SA.markovChain(initialization=seqAbove75[seqIdx], allowedPosIndices=indexGene, txtFile='asdf', modelFunc=seqToModelOutput, printing=False)
    pkl.dump(saOutputs, open(f'{args.outdir}/saOutputs_rank{seqIdx}_T{temperature:.3f}_max.pkl','wb'))

    
for tValue in tempRange:
    print(f'Temperature = {tValue}')
    saTemp(tValue)
    
# pool = Pool(processes=len(tempRange))
# pool.map(func=saTemp, iterable=tempRange)