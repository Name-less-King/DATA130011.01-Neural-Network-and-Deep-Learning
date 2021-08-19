import numpy as np 
import torch
import torchvision as tv 
from net.conv import extract_all_features
from helper import paintings,labels
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut

img_size = 224
myTransform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(img_size),
        tv.transforms.CenterCrop(img_size),
    ])

#进行特征提取
N = len(paintings)
x = np.zeros((N,54))
for i in range(N):
    path = paintings[i]
    img = tv.datasets.folder.default_loader(path)
    img = myTransform(img)
    img = img.numpy()

    _,x[i] = extract_all_features(img)

train_mask = np.array([al != -1 for al in labels])
val_mask = ~train_mask

loo = LeaveOneOut()
loo.get_n_splits(x[train_mask])
y = np.array(labels)

#输入特征集合，输出该集合的auc值作为评判特征的好坏
def mesure_of_feature_sets(x,y,fsets):
    #print('y',y)
    N,F = x.shape
    real_mask = np.array([ay==1 for ay in y])
    
    real_x = x[real_mask]
    assert real_x.shape[0] > 0 
    scores = np.zeros(N)
    cf_sets = []
    for f in fsets:
        cf = np.mean(real_x[:,f])
        cf_sets.append(cf) 
        df = x[:,f] - cf
        scores += df**2
    scores = -scores

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return auc

def decisions_of_feature_sets(x,y,fsets):
    
    N,F = x.shape
    real_mask = np.array([ay==1 for ay in y])
    
    real_x = x[real_mask]
    scores = np.zeros(N)
    cf_sets = []
    for f in fsets:
        cf = np.mean(real_x[:,f])
        cf_sets.append(cf) 
        df = x[:,f] - cf
        scores += df**2
    scores = -scores
    
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

    tnr = 1 - fpr 
    acc = (tpr + tnr) / 2
    thr = thresholds[np.argmax(acc)]

    return cf_sets,thr 

def predict_by_features(x,fsets, cf_sets,thr):
    N,F = x.shape
    scores = np.zeros(N)
    for cf,f in zip(cf_sets,fsets):
        df = x[:,f] - cf
        scores += df**2
    scores = -scores

    pred = np.zeros(N)
    for i in range(N):
        if scores[i] > thr:
            pred[i] = 1
        else:
            pred[i] = 0
    return pred

def select_features(x,y,num=5):
    N,F = x.shape
    features = []
    #贪心地选择使得AUC最大化的f
    for i in range(num):
        max_auc = 0
        max_auc_index = 0
        for f in range(F):
            if f in features:
                continue
            new_fsets = features + [f]
            auc  = mesure_of_feature_sets(x,y,new_fsets)
            if auc > max_auc:
                max_auc = auc 
                max_auc_index = f 
        features.append(max_auc_index)
    
    cf_sets, thr = decisions_of_feature_sets(x,y,features)
    return features,cf_sets,thr   

tot_num = 0
tot_correct = 0

tot_train_num = 0
tot_train_correct = 0

for train_index, test_index in loo.split(x[train_mask]):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #训练模型并且进行交叉验证

    features,cf_sets,thr = select_features(x_train,y_train)
    #print('thr',thr)
    pred = predict_by_features(x_test,features,cf_sets,thr)

    tot_num += 1
    tot_correct += np.sum(pred == y_test)
    
    #查看训练集预测结果
    pred_of_train = predict_by_features(x_train,features,cf_sets,thr)
    tot_train_correct += np.sum(pred_of_train == y_train) 
    tot_train_num += len(x_train)

train_acc = tot_train_correct / tot_train_num
test_acc = tot_correct / tot_num
print('train',train_acc,'test',test_acc)








    











