# Neural-Net-CRM-Classifier

An ANN classifer in Python using the KDD 2009 Cup small dataset. APIs include Keras, TensorFlow, Scikit-learn, and Hyperopt for random hyperparameter search.

Activations functions: ReLU, PReLU and Swish.
Optimizers: Adam, Nadam and SGD with nesterov momentum.

ACIDataPreProcessing.py - Removal of variables with low frequencies, imputation, etc.
ACI_ANN_Model.py - Running the hyperparameter search, selecting network with highest AUC score, re-running with best hyperparameters using checkpoint and early stopping.
