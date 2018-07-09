
# coding: utf-8

# # Neural Networks Project - Credit Risks Analysis
# 
# An experimental study about a neural network model aplication in a real world problem.
# 
# Neural Networks - Minister by Germano Vasconcelos
# 
# Team:  
# - Lucas Alves Rufno  
# - Rodrigo de Lima Oliveira  
# - Ullayne Fernandes Farias de Lima 
# - Vitor Jose da Silva Lima
# 
# ## Multilayer Perceptron Experiments 
# 
# Serie of experiments to evaluate the credit risks analysis using a statistical model
# 
# ### Imports:
# Relevant libraries to solve the problem

# In[ ]:


from sklearn.neural_network import MLPClassifier
from scipy.stats import ks_2samp as ksTest
from sklearn.utils import shuffle
from sklearn import metrics
import pandas as pd


# ### Read dataset:
# Read file as .h5 in Pandas with respective keys and describe data partially.

# In[ ]:


tr = pd.read_hdf("datasets/repeat/Train.h5", key='train')
va = pd.read_hdf("datasets/repeat/Validation.h5", key='validation')
te = pd.read_hdf("datasets/repeat/Test.h5", key='test')


# ### Modify dataset:
# Modifying the dataset to only 2 sets (Train and Test). The validation set is splitter in the algorithm. 

# In[ ]:


tr = tr.append(va)
tr = shuffle(tr)


# ### Trainning model:
# Generate the statistical model based

# In[ ]:


clf = MLPClassifier(
    hidden_layer_sizes=(500,),
    solver='sgd',
    activation='relu',
    learning_rate='adaptive',
    learning_rate_init=0.03,
    early_stopping=True,
    validation_fraction=0.1)
clf.fit(tr.iloc[:,:-1], tr['IND_BOM_1_1'])
rClass = clf.predict(te.iloc[:,:-1])
rProba = clf.predict_proba(te.iloc[:,:-1])[:,1]


# ### Evaluating model:
# Testing the statistical model

# In[ ]:


print('MSE:', metrics.mean_squared_error(te['IND_BOM_1_1'], rProba))
print('KS Test:', ksTest(te['IND_BOM_1_1'], rProba)[0])
print('ROC AUC:', metrics.roc_auc_score(te['IND_BOM_1_1'], rProba))
print('Accuracy:', metrics.accuracy_score(te['IND_BOM_1_1'], rClass))
print('Precision, Recall and FScore:')
print(metrics.precision_recall_fscore_support(te['IND_BOM_1_1'], rClass, average='binary')[:-1])
print('Confusion Matrix:')
print(metrics.confusion_matrix(te['IND_BOM_1_1'], rClass))

