import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier


class Model(object):
    def __init__(self, data_file, stack_file,feature_file,train_label):#solution.txt(id,lebel,content)  将传入的data文件前0.8有label，后面的0.2无lebel
        #stack_file为集成学习文件，先不用feature特征
        # 特征和微博数据X
        self.stack_file = stack_file  #文本特征
        self.features = pickle.load(open(feature_file, 'rb'))
        self.data_x = pickle.load(open(data_file, 'rb'))#训练集与测试集

        # 标签Y
        self.df = pd.read_csv(train_label, names=['id','label'],encoding='utf-8')#训练集
        self.label = LabelEncoder()  #标准化
        self.label.fit(self.df.label)
        self.df['y_label'] = self.label.transform(self.df.label)

#将tf-idf值用多种方法进行集成学习
    def stacking(self):
        X = self.data_x.content[:]#训练集加测试集
        print(X.shape,'gddg')
        #词粒度的tfidf值
        vectormodel = TfidfVectorizer(ngram_range=(1, 1), min_df=3, use_idf=False, smooth_idf=True, sublinear_tf=True,
                                      norm=False,token_pattern='(?u)\\b\\w+\\b')
        X = vectormodel.fit_transform(X)#词频矩阵
        print(X.shape)

        # 数据
        y = self.df.y_label  #取一列总共，训练集
        # y = Y[:len(Y)]  #训练集
        train_x = X[:len(y)]
        print(type(train_x), 'train_x ----------')
        test_x = X[len(y):].tocsc()  #Convert this matrix to Compressed Sparse Column format
        print(type(test_x),'test_x ----------')

        np.random.seed(0)

        n_folds = 5
        n_class = 4

        X = train_x  #训练集的tfidf
        y = y
        X_submission = test_x  #测试集tfidf

        skf = list(StratifiedKFold(y, n_folds))  #五折交叉验证  保证训练集中每一类的比例是相同的（尽量）根据y选择

        clfs = [
            # LogisticRegression(penalty='l1', n_jobs=-1, C=1.0),
            LogisticRegression(penalty='l2', n_jobs=-1, C=1.0),
            # RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            # RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='gini'),
            SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, max_iter=5, random_state=42),
            # ExtraTreesClassifier(n_estimators=200, n_jobs=-1, criterion='entropy')
        ]

        dataset_blend_train = np.zeros((X.shape[0], len(clfs) * n_class))
        dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs) * n_class))

        for j, clf in enumerate(clfs):#循环每个分类器
            print(j, clf)
            dataset_blend_test_j = np.zeros((X_submission.shape[0], n_class))
            for i, (train, test) in enumerate(skf):#已经分开了Train: [1 3 4 5 6 ] | test: [2]
                print('Fold ', i)
                X_train = X[train]
                y_train = y[train]
                X_test = X[test]
                y_test = y[test]
                clf.fit(X_train, y_train)
                y_submission = clf.predict_proba(X_test)
                dataset_blend_train[test, j * n_class:j * n_class + n_class] = y_submission
                dataset_blend_test_j += clf.predict_proba(X_submission)#将每次的预测相加，最后除以5
            dataset_blend_test[:, j * n_class:j * n_class + n_class] = dataset_blend_test_j[:, ] / n_folds  #转换为矩阵

        all_X_1 = np.concatenate((dataset_blend_train, dataset_blend_test), axis=0)  #预测值

        # xgboost
        temp = np.zeros((len(y), n_class))
        test = np.zeros((test_x.shape[0], n_class))
        test_x = test_x.tocsc()
        dtest = xgb.DMatrix(test_x)  #加载的数据存储在对象DMatrix中
        for tra, val in StratifiedKFold(y, 5, random_state=658):
            X_train = train_x[tra]
            y_train = y[tra]
            X_val = train_x[val]
            y_val = y[val]

            x_train = X_train.tocsc()
            x_val = X_val.tocsc()

            dtrain = xgb.DMatrix(x_train, y_train)
            dval = xgb.DMatrix(x_val)

            params = {
                "objective": "multi:softprob", #多分类的问题返回预测的概率，softmax返回预测类别
                "booster": "gblinear",
                "eval_metric": "merror",
                "num_class": 4,
                'max_depth': 3,# 构建树的深度，越大越容易过拟合
                'min_child_weight': 1.5,# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
                                        #假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                                        #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
                'subsample': 0.7, # 随机采样训练样本
                'colsample_bytree': 1,# 生成树时进行的列采样
                'gamma': 2.5,# 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                "eta": 0.01,# 如同学习率
                "lambda": 1,# 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                'alpha': 0,
                "silent": 1,#设置成1则没有运行信息输出，最好是设置为0.
            }
            watchlist = [(dtrain, 'train1')]#watchlist用于指定训练模型过程中用于监视的验证数据集
            model = xgb.train(params, dtrain, 2000, evals=watchlist,#early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
                              early_stopping_rounds=200, verbose_eval=200)
            result = model.predict(dval)
            temp[val] = result[:]#事先设置过形状

            res = model.predict(dtest)
            test += res
        test /= 5
        all_X_2 = np.concatenate((temp, test), axis=0)

        #############################################################################
        #############################################################################
        # merge
        all_X = np.concatenate((all_X_1, all_X_2), axis=1)
        pickle.dump(all_X, open(self.stack_file, 'wb'))  #6种分类器与集成学习  写入文件stack.txt

    def concat_features(self,outfeatures = None):#与另一个特征文件连接
        print('concat features...')
        all_X = pickle.load(open(self.stack_file, 'rb'))
        myfeature = self.features.drop(['id'], axis=1).as_matrix() #去掉id将所有features取出来
        # train1+test set
        self.all_X = np.concatenate((all_X, myfeature), axis=1)#行对应
        if outfeatures:
            featureslist = pickle.load(open(outfeatures, 'rb'))
            for fea in featureslist:
                self.all_X = np.concatenate((self.all_X, fea), axis=1)
        # train1 set
        self.X = self.all_X[:self.df.shape[0]]  #训练集的所有x
        self.y = self.df.y_label

        print('特征维数为{}维'.format(self.X.shape[1]))

    def fit_transform(self, result_label):#放进结果文件中
        print('bagging...')
        # self.all_X = pickle.load(open(self.stack_file, 'rb'))
        # self.X = self.all_X[:int(self.df.shape[0] * 0.8)]  # 训练集的所有x
        # self.y = self.df.y_label[:int(len(self.df.y_label) * 0.8)]
        n = 8  #分为8份
        score = 0
        pres = []
        i = 1
        for tra, val in StratifiedShuffleSplit(self.y, n, test_size=0.2, random_state=233):#random_state控制是将样本随机打乱
            print('run {}/{}'.format(i, n))
            i += 1

            X_train = self.X[tra]
            y_train = self.y[tra]
            X_val = self.X[val]
            y_val = self.y[val]

            dtrain = xgb.DMatrix(X_train, y_train)
            dval = xgb.DMatrix(X_val, y_val)
            dtest = xgb.DMatrix(self.all_X[self.df.shape[0]:])

            params = {
                "objective": "multi:softmax",
                "booster": "gbtree",
                "eval_metric": "merror",
                "num_class": 4,
                'max_depth': 3,
                'min_child_weight': 1.5,
                'subsample': 0.7,
                'colsample_bytree': 1,
                'gamma': 2.5,
                "eta": 0.01,
                "lambda": 0.8,
                'alpha': 0.1,
                "silent": 1,
            }
            watchlist = [(dtrain, 'train1'), (dval, 'eval')]#watchlist用于指定训练模型过程中用于监视的验证数据集
            bst = xgb.train(params, dtrain, 2000, evals=watchlist,#判断是否相同
                            early_stopping_rounds=200, verbose_eval=False)
            # score += bst.best_score#计算正确率

            pre = bst.predict(dtest)
            pres.append(pre)

        score /= n
        score = 1 - score
        print('*********************************************')
        print('*********************************************')
        print("******分类平均准确率为{}**************".format(score))
        print('*********************************************')
        print('*********************************************')

        # vote
        pres = np.array(pres).T.astype('int64')
        pre = []
        for line in pres:#预测了8次，取出现次数最多的类别
            pre.append(np.bincount(line).argmax())#它大致说bin的数量比x中的最大值大1，每个bin给出了它的索引值在x中出现的次数。

        csv_file_id = open('../data/valid data/resolution_valid.txt', encoding='utf-8')
        ids = []
        for line in csv_file_id:
            id = line.strip().split(',')[0]
            ids.append(id)

        result = pd.DataFrame(pre, columns=['label'])
        result['label'] = self.label.inverse_transform(result.label)
        result.insert(0,'id',ids)

        result.to_csv(result_label, index=None,header=None)
        print('result saved!')
