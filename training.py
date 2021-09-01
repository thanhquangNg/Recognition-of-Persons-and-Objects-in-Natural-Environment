import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
data = pd.read_csv('sum1.csv')
data_test = pd.read_csv('sample1.csv')
y_test = data_test.iloc[:,1].values
# Read dataset to pandas dataframe
#dataset = pd.read_csv(url, names=names)
data_t=data
#print()

X = data_t.iloc[:, :-1].values
y = data_t.iloc[:, 3].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
filename = 'knn_python.sav'
pickle.dump(classifier, open(filename,'wb'))
feed_rms = []
feed_var = []
feed_fft = []

sample_data = data_test.iloc[:,1].values
print(sample_data)

rms = np.sqrt(np.mean(sample_data**2))
var = np.var(sample_data)
cov = np.cov(sample_data, bias=True)
fft = max(abs(np.fft.fft(np.abs(sample_data))))

feed_rms.append(rms)
feed_var.append(var)
feed_fft.append(fft)

feed_data = feed_rms + feed_var + feed_fft
print(feed_data)
y_pred = classifier.predict(X_test)
print(y_pred)

'''from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
'''
'''
# Plot non-normalized confusion matrix

disp = plot_confusion_matrix(classifier, X_train, y_train)
disp.ax_.set_title("Confusion Matrix for Training Result")
disp2 = plot_confusion_matrix(classifier, X_test, y_test)
disp2.ax_.set_title("Confusion Matrix for Testing Result")


plt.show()