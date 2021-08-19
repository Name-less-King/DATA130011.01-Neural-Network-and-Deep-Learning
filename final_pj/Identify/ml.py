import numpy as np
import pickle

import sklearn
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import tree

from feature_extract import Tight_frame_classifier

Tight_frame_model = pickle.load(open('model.pkl','rb'))
Feature_index = Tight_frame_model.best_feature()

Train_set = pickle.load(open('Train_set.pkl','rb'))
Test_set = pickle.load(open('Test_set.pkl','rb'))
Train_label = pickle.load(open('Train_label.pkl','rb'))
Test_label = pickle.load(open('Test_label.pkl','rb'))
Disputed = pickle.load(open('tight_frame_D.p','rb'))

# （可选）这里我们可以先做一个 Forward-stagewise
Train = Train_set[:,Feature_index]
Test = Test_set[:,Feature_index]
Disputed = Disputed[:,Feature_index]


# 测试不同机器学习方法
# clf = SVC(kernel='linear')
# knn = neighbors.KNeighborsClassifier()

# 据reference 对于gabor wavelet 我们需要使用二次核
# clf = SVC(kernel='poly',degree=2)

DT = tree.DecisionTreeClassifier()

# clf.fit(Train, Train_label)
# knn.fit(Train,Train_label)
DT.fit(Train,Train_label)

# 测试模型,我们模型的一大任务是判断有争议的画作
# pred_D = clf.predict(Disputed)
# pred_D = knn.predict(Disputed)
pred_D = DT.predict(Disputed)
print(pred_D)

# 另一任务是鉴定准确率是否足够
pred = clf.predict(Test)
pred = knn.predict(Test)
pred = DT.predict(Test)
precision = sklearn.metrics.precision_score(Test_label, pred)
recall = sklearn.metrics.recall_score(Test_label, pred)
accuracy = sklearn.metrics.accuracy_score(Test_label, pred)

TP = 0; TN = 0
for i in range(len(Test_label)):
    if(pred[i] == Test_label[i] == 1):
        TP += 1
    elif(pred[i] == Test_label[i] == 0):
        TN += 1
TPR = TP / sum(Test_label)
TNR = TN / (len(Test_label)-sum(Test_label))

print('\naccuracy: %.2f%%' % (100 * accuracy))
print('TPR: %.2f%%\nTNR: %.2f%%' % (100 * TPR, 100 * TNR))
print('precision: %.2f%%\nrecall: %.2f%%' % (100 * precision, 100 * recall))



