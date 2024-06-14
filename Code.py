# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 22:47:56 2024

@author: Junaid
"""


import pandas as pd


import pandas as pd
import numpy as np


from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("Data.csv")

parameter = (25,4165) 

y = data["Label"]

x = data.drop("Label" ,axis = 1)



import numpy as np

def add_gaussian_noise(signal, mean=0, std=0.1):
    """
    Add Gaussian noise to a signal.
    
    Args:
    - signal: 1D numpy array representing the signal.
    - mean: Mean of the Gaussian distribution (default: 0).
    - std: Standard deviation of the Gaussian distribution (default: 0.1).
    
    Returns:
    - noisy_signal: Signal with added Gaussian noise.
    """
    noise = np.random.normal(mean, std, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

def add_gaussian_noise_to_dataset(dataset, labels, mean=0, std=0.1, num_signals=100):
    """
    Add Gaussian noise to all signals in a dataset and create a total of num_signals signals.
    
    Args:
    - dataset: 2D numpy array representing the dataset where each row is a signal.
    - labels: 1D numpy array representing the labels of original signals.
    - mean: Mean of the Gaussian distribution (default: 0).
    - std: Standard deviation of the Gaussian distribution (default: 0.1).
    - num_signals: Total number of signals to create (default: 100).
    
    Returns:
    - noisy_dataset: Dataset with added Gaussian noise to all signals and a total of num_signals signals.
    - noisy_labels: Labels for the new signals.
    """
    num_original_signals = dataset.shape[0]
    num_signals_to_create = num_signals - num_original_signals
    noisy_dataset = np.zeros((num_signals, dataset.shape[1]))
    
    # Add Gaussian noise to original signals
    for i in range(num_original_signals):
        noisy_dataset[i, :] = add_gaussian_noise(dataset[i, :], mean, std)
    
    # Create additional signals by adding Gaussian noise to existing signals
    for i in range(num_original_signals, num_signals):
        random_index = np.random.randint(0, num_original_signals)  # Choose a random original signal
        noisy_dataset[i, :] = add_gaussian_noise(dataset[random_index, :], mean, std)
    
    # Assign labels to new signals
    original_labels = labels[:num_original_signals]
    new_labels = np.random.choice(original_labels, num_signals_to_create)
    noisy_labels = np.concatenate((original_labels, new_labels))
    
    return noisy_dataset, noisy_labels

# Example usage:
# Assuming 'signals' is your dataset with shape (25, 4166) and 'labels' is the corresponding label column
# noisy_signals, noisy_labels = add_gaussian_noise_to_dataset(signals, labels, num_signals=100)


#noisydata = add_gaussian_noise_to_dataset(x,y,mean=0, std=0.001, num_signals=100)

#x_train, x_test, y_train, y_test = train_test_split(x, y)



gaussian_noise = np.random.normal(0, 0.001, parameter)

gaussian_noise2 = np.random.normal(0, 0.001, parameter)

gaussian_noise3 = np.random.normal(0, 0.001, parameter)

gaussian_noise4 = np.random.normal(0, 0.001, parameter)


x1 = x+gaussian_noise

x2 = x + gaussian_noise2

x3 = x + gaussian_noise3

 

check = x

check["label"] = y

check2 = x1

check2["label"] = y

check3 = x2

check3["label"] = y

check4 = x3

check4["label"] = y


total1 = pd.concat([check, check2, check3,check4], ignore_index=True)

labels = total1["label"]
train = total1.drop("label", axis = 1)


x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.5,stratify=labels, random_state=42)

model = AdaBoostClassifier()

model.fit(x_train, y_train) 

y_pred = model.predict(x_test)




from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
    

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


import pickle
filename = 'AdaBoost92%Accuracy.sav'
pickle.dump(model, open(filename, 'wb'))