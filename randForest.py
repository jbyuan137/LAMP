# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pandas as pd
import numpy as np
import pickle
import warnings
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report, precision_score, roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from tqdm import tqdm


p = argparse.ArgumentParser(description='Parse the arguments')
p.add_argument('-b', '--onlyBERT', type=int, help='Use only BERT features, 1 if True, 0 if False', required=True)
p.add_argument('-r', '--useRandom', type=int, help='Use additional random features, 1 if True, 0 if False', required=True)
p.add_argument('-c', '--cv', type=int, help='Number of cross validation folds', required=True)
p.add_argument('-m', '--metric', type=str, help='Number of cross validation folds', required=True)
args = p.parse_args()

squareSize = 7.5

onlyBERT = args.onlyBERT
useRandom = args.useRandom
metricDict={'bacc':'balanced_accuracy',
            'f1':'f1',
            'recall':'recall'}

metricString = str(args.metric)
metricType = metricDict[args.metric]

folds = args.cv


########################################     LOADING/PROCESSING DATA     ########################################

# Load the data

df = pd.read_csv('pathToDataFile.csv')

# Define the bins and labels of predictive Amp value
bins = [-0.1, 99.99, 100.1] 
# labels = [0, 100]
labels = [0, 100]

# Categorize the data
df['categorized_percentage'] = pd.cut(df['Amp'], bins=bins, labels=labels, right=False)
df['categorized_percentage'] = df['categorized_percentage'].astype(float)

# Create maxEmb DataFrame including 'primer' and 'ct'
maxEmb = pd.DataFrame(np.round(pd.DataFrame(df['maxEmbed'].to_list()).values, decimals=3))
maxEmb['categorized_percentage'] = df['categorized_percentage']
maxEmb['Primer'] = df['Primer']
maxEmb['GC_Content'] = df['GC_Content']
maxEmb['Gene'] = df['Gene']
maxEmb['Mut_Amount'] = df['Mut_Amount']
maxEmb['Sub_Type'] = df['Sub_Type']

# One-hot encode categorical features
maxEmb = pd.get_dummies(maxEmb, columns=['Primer', 'Gene', 'Sub_Type'], drop_first=False)

# Create one_hot_columns list
Primer_columns = [col for col in maxEmb.columns if isinstance(col, str) and 'Primer_' in col]
GC_columns = [col for col in maxEmb.columns if isinstance(col, str) and 'GC_' in col]
Gene_columns = [col for col in maxEmb.columns if isinstance(col, str) and 'Gene_' in col]
Mut_columns = [col for col in maxEmb.columns if isinstance(col, str) and 'Mut_' in col]
Sub_columns = [col for col in maxEmb.columns if isinstance(col, str) and 'Sub_' in col]

# Combine both lists into one_hot_columns
one_hot_columns = Primer_columns + Gene_columns + Sub_columns
nonbert_numerical_columns = ['GC_Content','Mut_Amount'] 

# Ensure the columns are of integer type
maxEmb[one_hot_columns] = maxEmb[one_hot_columns].astype(int)

# Perform downstream analysis on numerical (not one-hot encoded variables)
x = maxEmb.drop(['categorized_percentage'] + one_hot_columns+nonbert_numerical_columns, axis=1).values
y = maxEmb['categorized_percentage'].values

bertColumns = [f'BERT{i+1}' for i in range(x.shape[1])]

bertDF = pd.DataFrame(data=x, columns=bertColumns)

labeledBertDF = pd.concat([bertDF, 
                     pd.DataFrame(data=y, columns=['categorized_percentage']), 
                     maxEmb[nonbert_numerical_columns].reset_index(drop=True),                           
                     maxEmb[one_hot_columns].reset_index(drop=True)], axis=1)

labeledBertDF['stratify_col'] = labeledBertDF['categorized_percentage'].astype(str) + '_' + df['Gene'].astype(str)

random_state = 7722
print('Seed =', random_state)
np.random.seed(seed=random_state)


# Split the data into training and testing sets (90% train, 10% test), keeping equal proportion of genes that amplify
X_train, X_test, y_train, y_test = train_test_split(labeledBertDF.drop(['categorized_percentage'], axis=1),
                                                    labeledBertDF['categorized_percentage'],
                                                    test_size=0.1,
                                                    random_state=random_state,
                                                    stratify=labeledBertDF['stratify_col'])

X_train_bert = X_train[bertColumns]
X_test_bert = X_test[bertColumns]

scaledFitBERT = StandardScaler().fit(X_train_bert)
X_train_bertz = scaledFitBERT.transform(X_train_bert)
X_test_bertz = scaledFitBERT.transform(X_test_bert)

variance = []
for nTop in range(1,100):
    pca = PCA(n_components=nTop, random_state=random_state)
    pcaFit = pca.fit(X_train_bertz)
    variance.append(sum(pcaFit.explained_variance_ratio_))

plt.figure(figsize=(squareSize,squareSize))
plt.plot(variance)

variance = []

plt.xlabel('Number of principal components', fontsize=14)  
plt.ylabel('Cumulative Variance Ratio', fontsize=14)   
# plt.title('', fontsize=14)  
# plt.legend(loc='lower right', fontsize=14) 
for nTop1 in range(1,100):
    pca = PCA(n_components=nTop1, random_state=random_state)
    pcaFit = pca.fit(X_train_bertz)
    variance.append(sum(pcaFit.explained_variance_ratio_))
    if sum(pcaFit.explained_variance_ratio_) >= 0.80:
        break
print(nTop1, variance[nTop1-1])
nTop1 = 7

pca = PCA(n_components=nTop1, random_state=random_state)
pcaFit = pca.fit(X_train_bertz)

X_train_pca = pcaFit.transform(X_train_bertz)
X_test_pca = pcaFit.transform(X_test_bertz)

X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

if onlyBERT == True:
    bertString='bert_features'
    X_train_df = pd.concat([X_train_pca_df, 
                        pd.DataFrame(data=y_train, columns=['categorized_percentage']).reset_index(drop=True)],
                        axis=1)

    X_test_df = pd.concat([X_test_pca_df, 
                        pd.DataFrame(data=y_test, columns=['categorized_percentage']).reset_index(drop=True)], 
                        axis=1)
elif onlyBERT == False:
    bertString='all_features'
    X_train_df = pd.concat([X_train_pca_df, 
                        pd.DataFrame(data=y_train, columns=['categorized_percentage']).reset_index(drop=True),
                        X_train[nonbert_numerical_columns].reset_index(drop=True),
                        X_train[one_hot_columns].reset_index(drop=True)], axis=1)

    X_test_df = pd.concat([X_test_pca_df, 
                        pd.DataFrame(data=y_test, columns=['categorized_percentage']).reset_index(drop=True),
                        X_test[nonbert_numerical_columns].reset_index(drop=True),
                        X_test[one_hot_columns].reset_index(drop=True)], axis=1)
elif onlyBERT == 2:
    bertString='categorical_features'
    X_train_df = pd.concat([pd.DataFrame(data=y_train, columns=['categorized_percentage']).reset_index(drop=True),
                        X_train[nonbert_numerical_columns].reset_index(drop=True),
                        X_train[one_hot_columns].reset_index(drop=True)], axis=1)

    X_test_df = pd.concat([pd.DataFrame(data=y_test, columns=['categorized_percentage']).reset_index(drop=True),
                        X_test[nonbert_numerical_columns].reset_index(drop=True),
                        X_test[one_hot_columns].reset_index(drop=True)], axis=1)

print("X_train_df shape =", X_train_df.shape)
print("X_train_df columns names:", X_train_df.columns)

featureCols = [term for term in X_train_df.columns if 'categorized_percentage' not in term]


train_data = X_train_df[featureCols]
train_data['target'] = X_train_df['categorized_percentage']

X_test_df = X_test_df.drop('categorized_percentage', axis=1) 

if useRandom == True:
    randomString = '_with_random'
    train_data['random_numerical'] = np.random.normal(loc=0, scale=1, size=len(train_data))
    train_data['random_categorical'] = np.random.choice([0,1], size=len(train_data), p=[0.5,0.5])
    X_test_df['random_numerical'] = np.random.normal(loc=0, scale=1, size=len(X_test_df))
    X_test_df['random_categorical'] = np.random.choice([0,1], size=len(X_test_df), p=[0.5,0.5])
elif useRandom == False:
    randomString = ''


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


X_train = X_train_df
y_train = y_train_df
X_test = X_test_df

print("X_train columns:", X_train.columns)
print("X_test columns:", X_test.columns)


# Calculate class weights based on class frequencies
class_weights_freq = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_freq_dict = {label: weight for label, weight in zip(np.unique(y_train), class_weights_freq)}

# Display the calculated class weights based on frequencies
print("Class Weights based on frequencies:", class_weights_freq_dict)

# Calculate sample weights based on class weights from frequencies
sample_weights_freq = np.array([class_weights_freq_dict[label] for label in y_train])


########################################     TRAINING MODEL     ######################################## 


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90, 100],  
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': [2,3,4,5],  
    'min_samples_split': [2, 3, 4, 5], 
    'min_samples_leaf': [2,3,4,5]  
}

# Initialize the gbclassifier
rf_weights_freq = RandomForestClassifier(random_state=random_state)

stratifiedK = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=rf_weights_freq, param_grid=param_grid, scoring=metricType, cv=stratifiedK, return_train_score=True, n_jobs=-1)

# Fit the GridSearchCV on the training set
grid_search.fit(X_train, y_train, sample_weight=sample_weights_freq)


# Initialize variables to track the best parameters and the minimum difference
best_params = None
min_diff = 100.0

# Store results for each parameter combination
results = []

# Evaluate each parameter combination
best_score = 0
for params, test_score, train_score in tqdm(zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['mean_train_score'])):
    # Fit the model with the current parameters
    differences = train_score - test_score

    if differences < min_diff:
    # Update the best parameters if the current overall difference is smaller
        best_score = test_score
        best_diff = differences
        best_params = params

# Print the best parameters found
print('# of PCs', nTop1)
print("Best test score found: ", best_score)
print("Best difference found: ", best_diff)
print("Best parameters found: ", best_params)


# Evaluate the best model on the training set
best_model = RandomForestClassifier(**best_params, random_state=random_state)
best_model.fit(X_train, y_train, sample_weight=sample_weights_freq)

with open(f'modelFiles/{metricString}/rf_cv{folds}_{bertString}{randomString}.pkl', 'wb') as file:
    pickle.dump(best_model, file)

    
########################################     EVALUATING MODEL     ######################################## 


def geneFormat(string):
    splitString = string.split('_')
    gene = splitString[-1]
    return('Gene ' + r'$\it{' + gene + '}$')

def primerFormat(string):
    splitString = string.split('_')
    primers = ','.join(splitString[1:])
    return(f'{primers} Mut.')

def subFormat(string):
    splitString = string.split('_')
    mutType = ', '.join(splitString[2:])
    return(mutType.replace('substitution','Sub.').replace('deletion','Del.').replace('none','No Mut.'))

def randFormat(string):
    return(string.replace('Random_categorical','Rand. One-Hot').replace('Random_numerical','Rand. Num').replace('none','No Mut.'))

bestest_model = pickle.load(open(f'modelFiles/{metricString}/rf_cv{folds}_{bertString}{randomString}.pkl','rb'))

pred_train = bestest_model.predict(X_train)
accuracy_train = accuracy_score(y_train, pred_train)
balanced_train = balanced_accuracy_score(y_train, pred_train)
precision_train = precision_score(y_train, pred_train, average='weighted', zero_division=0)

# Print the evaluation metrics for the training set
print(f"Training Set - Accuracy: {accuracy_train:.3f}, Balanced Accuracy: {balanced_train:.3f}, Precision: {precision_train:.3f}")

# Generate and print the classification report for the training data
print("\nClassification Report for Training Data:")
print(classification_report(y_train, pred_train, zero_division=0))

# Predict on the test set
pred_test = bestest_model.predict(X_test)

# Evaluate the model on the test set
accuracy_test = accuracy_score(y_test, pred_test)
balanced_test = balanced_accuracy_score(y_test, pred_test)

precision_test = precision_score(y_test, pred_test, average='weighted', zero_division=0)

# Print the evaluation metrics for the test set
print(f"Test Set - Accuracy: {accuracy_test:.4f},  Balanced Accuracy: {balanced_test:.3f}, Precision: {precision_test:.4f}")

# Generate and print the classification report for the test data
print("\nClassification Report for Test Data:")
print(classification_report(y_test, pred_test, zero_division=0))

# Generate and print the classification report for the test data
print("\nClassification Report for Test Data:")
print(classification_report(y_test, pred_test, zero_division=0))

# Calculate predicted probabilities for the test set
y_scores = bestest_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (100.0)

# Store true labels for the test set
y_test_labels = y_test  # True labels for the test set

# Calculate precision and recall values
precision, recall, thresholds = precision_recall_curve(y_test_labels, y_scores, pos_label=100.0)

# Calculate the AUC for the Precision-Recall curve
pr_auc = auc(recall, precision)

# Now you can use y_test_labels and y_scores to plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_labels, y_scores, pos_label=100.0)
roc_auc = auc(fpr, tpr)

# Prepare data for export
output_data = {
    "Predicted Probabilities": y_scores,
    "True Labels": y_test_labels,
    "Precision Values": precision,
    "Recall Values": recall,
    "AUC for Precision-Recall Curve": pr_auc,
    "False Positive Rate": fpr,
    "True Positive Rate": tpr,
    "AUC": roc_auc
}

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentages
    print(cm_percent)

    plt.figure(figsize=(squareSize, squareSize))
    sns.heatmap(cm_percent, cmap='Blues', cbar=False,
                xticklabels=['0.0', '100.0'], yticklabels=['0.0', '100.0'],
                annot=False)  # Disable default annotations
    plt.title(title, fontsize=14)  # Title font size
    plt.xlabel('Predicted Label', fontsize=14)  # X-axis label
    plt.ylabel('True Label', fontsize=14)  # Y-axis label
    plt.xticks(fontsize=14)  # X-axis ticks
    plt.yticks(fontsize=14)  # Y-axis ticks

    # Add percentage sign to annotations manually
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            if i == j:
                plt.text(j + 0.5, i + 0.5, f'{cm_percent[i, j]:.2f}%', 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            else:
                plt.text(j + 0.5, i + 0.5, f'{cm_percent[i, j]:.2f}%', 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='black')

# Plot confusion matrix for training data
print("Train Data Confusion Matrix")
plot_confusion_matrix(y_train, pred_train, title='Confusion Matrix for Training Data')

# Save the confusion matrix plot for training data as PDF
plt.savefig(f'Figures/{metricString}/rf_confusion_train_cv{folds}_{bertString}{randomString}.svg', format='svg', bbox_inches='tight', dpi=300)

# Plot confusion matrix for test data

print("\nTest Data Confusion Matrix")
plot_confusion_matrix(y_test, pred_test, title='Confusion Matrix for Test Data')
plt.savefig(f'Figures/{metricString}/rf_confusion_test_cv{folds}_{bertString}{randomString}.svg', format='svg', bbox_inches='tight', dpi=300)

# bestest_model.fit(X_train, y_train, sample_weight=sample_weights_freq)

# Predict probabilities for the test set
y_prob = bestest_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=100)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(squareSize, squareSize))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=14)

# Set tick label font sizes
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

plt.grid()
plt.savefig(f'Figures/{metricString}/rf_roc_test_cv{folds}_{bertString}{randomString}.svg', format='svg', bbox_inches='tight', dpi=300)

# Predict probabilities for the test set
y_prob = bestest_model.predict_proba(X_test)[:, 1]

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label=100)

# Calculate AUC
pr_auc = auc(recall, precision)

# Plotting the Precision-Recall curve
plt.figure(figsize=(squareSize, squareSize))
plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = {:.2f})'.format(pr_auc))
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=14)
plt.plot([1, 0], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.legend(loc='lower left', fontsize=14)

# Set tick label font sizes
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.savefig(f'Figures/{metricString}/rf_prc_test_cv{folds}_{bertString}{randomString}.svg', format='svg', bbox_inches='tight', dpi=300)


importances = bestest_model.feature_importances_  # Use the best model's feature importances

# Create a list of feature names

if onlyBERT == True:
    feature_names = ['PC' + str(i) for i in range(1, nTop1+1)]
elif onlyBERT == False:
    feature_names = ['PC' + str(i) for i in range(1, nTop1+1)] + Primer_columns + GC_columns + Gene_columns + Mut_columns + Sub_columns
elif onlyBERT == 2:
    feature_names = Primer_columns + GC_columns + Gene_columns + Mut_columns + Sub_columns

if useRandom == True:
    feature_names += ['Random_numerical','Random_categorical']
if useRandom == False:
    feature_names += []

# print(feature_renamed)

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Group by main categories and sum their importances
importance_df['Main_Category'] = importance_df['Feature'].str.split('_').str[0]  # Get the main category from the feature name
aggregated_importance_df = importance_df.groupby('Main_Category', as_index=False).agg({'Importance': 'sum'})

renamedAggregated = []
for term in aggregated_importance_df['Main_Category'].values:
    if term == 'Mut':
        renamedAggregated.append('# of Mut.')
    elif term == 'GC':
        renamedAggregated.append('% G or C')
    elif term == 'Sub':
        renamedAggregated.append('Mut. Type')
    else:
        renamedAggregated.append(term)
aggregated_importance_df['Main_Category'] = renamedAggregated

# Sort the DataFrame by importance
aggregated_importance_df = aggregated_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(squareSize, squareSize))
sns.barplot(x='Importance', y='Main_Category', data=aggregated_importance_df, color='grey')

# Set title and labels with specified font sizes
# plt.title('Aggregated Feature Importances', fontsize=14)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature Categories', fontsize=14)

# Adjust x-axis tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=12)
plt.savefig(f'Figures/{metricString}/rf_aggregated_gini_importance_cv{folds}_{bertString}{randomString}.svg', format='svg', bbox_inches='tight', dpi=300)

if onlyBERT == True:
    feature_names = ['PC' + str(i) for i in range(1, nTop1+1)]
elif onlyBERT == False:
    feature_names = ['PC' + str(i) for i in range(1, nTop1+1)] + Primer_columns + GC_columns + Gene_columns + Mut_columns + Sub_columns

if useRandom == True:
    feature_names += ['Rand. Num.','Rand. One-Hot']
if useRandom == False:
    feature_names += []


feature_renamed = []
for item in feature_names:
    if 'Primer' in item:
        feature_renamed.append(primerFormat(item))
    elif 'Gene_' in item:
        feature_renamed.append(geneFormat(item))
    elif 'Sub' in item:
        feature_renamed.append(subFormat(item))
    elif 'GC'  in item:
        feature_renamed.append('% G or C')
    elif 'Mut_Amount' in item:
        feature_renamed.append('# of Mut.')
    elif 'Random' in item:
        feature_renamed.append(randFormat(item))
    else:
        feature_renamed.append(item)


if useRandom == True:
    sepRandomString = '_with_separated_random'


    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Group by main categories and sum their importances
    importance_df['Main_Category'] = importance_df['Feature'].str.split('_').str[0]  # Get the main category from the feature name
    aggregated_importance_df = importance_df.groupby('Main_Category', as_index=False).agg({'Importance': 'sum'})

    renamedAggregated = []
    for term in aggregated_importance_df['Main_Category'].values:
        if term == 'Mut':
            renamedAggregated.append('# of Mut.')
        elif term == 'Sub':
            renamedAggregated.append('Mut. Type')
        else:
            renamedAggregated.append(term)
    aggregated_importance_df['Main_Category'] = renamedAggregated

    # Sort the DataFrame by importance
    aggregated_importance_df = aggregated_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(squareSize, squareSize))
    sns.barplot(x='Importance', y='Main_Category', data=aggregated_importance_df, color='grey')

    # Set title and labels with specified font sizes
    # plt.title('Aggregated Feature Importances', fontsize=14)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature Categories', fontsize=14)

    # Adjust x-axis tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.savefig(f'Figures/{metricString}/rf_aggregated_gini_importance_cv{folds}_{bertString}{sepRandomString}.svg', format='svg', bbox_inches='tight', dpi=300)

separated_importance_dict = {'Importance': importances, 'Features': feature_renamed}
separated_importance_df = pd.DataFrame(separated_importance_dict)
separated_importance_df = separated_importance_df.sort_values(by = 'Importance', ascending = False)

plt.figure(figsize=(2*squareSize, squareSize/2))
plt.bar(x = separated_importance_df['Features'], height = separated_importance_df['Importance'])

# Set title and labels with specified font sizes
# plt.title('Aggregated Feature Importances', fontsize=14)
plt.ylabel('Importance', fontsize=14)
# plt.xlabel('Feature Categories', fontsize=14)

# Adjust x-axis tick label sizes
plt.xticks(ticks = np.arange(len(separated_importance_df['Features'])), labels = separated_importance_df['Features'], rotation = 90, fontsize=11)

plt.yticks(fontsize=14)
plt.xlim(-0.5,len(feature_names))
plt.savefig(f'Figures/{metricString}/rf_separated_gini_importance_cv{folds}_{bertString}{randomString}.svg', format='svg', bbox_inches='tight', dpi=300)
