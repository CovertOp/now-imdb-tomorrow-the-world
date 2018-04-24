import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#names = ['bebe','nina','rebecca','tyra','raven','juju','raja','manilla','alexis','sharon','chad','phiphi','jinkx','alaska','roxxxy','bianca','adore','courtney','violet','ginger','pearl','bob','kimchi','naomi','sasha','peppermint','shea','trinity']
dataset = pd.read_csv('dataset.csv')
gurls = pd.read_csv('new_gurls.csv')

'''
dataset.plot(kind='box', subplots=True, layout=(4,7), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()
'''

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
