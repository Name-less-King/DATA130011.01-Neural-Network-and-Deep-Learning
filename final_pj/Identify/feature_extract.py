import numpy as np
import pickle
from sklearn.model_selection import LeaveOneOut

import utils.forward_stagewise as fs

Feature_max_size = 96

class Tight_frame_classifier(object):
    def __init__(self,best_size,Feature,mean_train,SD,Center,Threshold):
        self.selected_feature = Feature
        self.mean_train = mean_train
        self.std_train = SD
        self.center_true = Center
        self.thres = Threshold
        self.best_size = best_size

    def best_feature(self):
        return self.selected_feature[:self.best_size]

    def predict(self,test,feature_size = None):
        if (feature_size == None) :
            feature_size = self.best_size
        if (type(test[0]) == np.float):
            test = [test]
        prediction = []
        sd = self.std_train[feature_size-1]
        center = self.center_true[feature_size-1]
        selected_feature = self.selected_feature[:feature_size]
        thres = self.thres[feature_size-1]
        for ite in test:
            T = (ite-self.mean_train) / sd
            T = T[selected_feature]
            center_true_temp = center[selected_feature]
            dist = (((T - center_true_temp) ** 2).sum()) ** 0.5
            prediction += [dist < thres]
        return prediction

if __name__ == '__main__':
    Train_label = pickle.load(open('Train_label.pkl','rb'))
    Train_set = pickle.load(open('Train_set.pkl','rb'))
    Feature_size = Train_set.shape[1]

    Center = np.zeros([Feature_max_size,Feature_size])
    SD = np.zeros([Feature_max_size,Feature_size])
    Threshold = []
    Feature = []
    Score = []

    for feature_size in range(Feature_max_size):
        print('进行特征维度：',feature_size+1)
        potential_feature = [i for i in range(96) if i not in Feature]
        potential_center = np.zeros([len(potential_feature),Feature_size])
        potential_SD = np.zeros([len(potential_feature), Feature_size])
        potential_thre = []
        score = []
        for ite in range(len(potential_feature)):
            point = 0
            selected_feature = Feature + [int(potential_feature[ite])]
            print(selected_feature)
            for i in range(len(Train_set)):
                vali = Train_set[i]
                vali_label = Train_label[i]
                train = np.column_stack((Train_set[:i, :].T, Train_set[i + 1:, :].T)).T
                train_label = np.append(Train_label[:i], Train_label[i + 1:])
                #选择特征
                selected_feature, auc_select, mean_train, std_train, center_true, thres \
                    = fs.select_feature(train,train_label,feature_size,selected_feature)
                potential_center[ite] = center_true
                potential_SD[ite] = std_train
                potential_thre += [thres]

                predict = fs.predict([vali], selected_feature, mean_train, std_train, center_true, thres)
                point += (predict[0] == vali_label)

            print(point[0])
            score += [point[0]]

        
        best_index = score.index(max(score))
        Score += [max(score)]
        print('维度分数:',Score)
        Feature += [potential_feature[best_index]]
        Threshold += [potential_thre[best_index]]
        SD[feature_size] = potential_SD[best_index]
        Center[feature_size] = potential_center[best_index]
    
    mean_train = Train_set.mean(axis=0)
    best_size = Score.index(max(Score)) + 1
    classifier = Tight_frame_classifier(best_size, Feature, mean_train, SD, Center, Threshold)
    Model_file = open('model.pkl','wb')
    pickle.dump(classifier,Model_file)
