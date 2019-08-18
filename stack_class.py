import os
import keras
import pickle
from scipy import sparse
from keras.layers import Conv2D, MaxPooling2D
from scipy.sparse.csr import csr_matrix
from  keras.models import Sequential
# from keras.layers.core import Dense,Flatten,Dropout
from keras.layers import Dense, Dropout, Flatten
# from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from 评测.src.keras_helper import ModelCheckpointPlus
from 评测.src.models import StackEnsemble
from 评测.src.models import MCNN2
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

def load_v2():
    return pickle.load(open('../data/feature/features.v2.pkl','rb'))

ids,ys,f_train,f_test,f_content=load_v2()  #传入feature
f_text,f_tongji=f_train
f_text_test,f_tongji_test=f_test
ReTrain=False
y_class=ys.astype(int)#神马意思
print(ys.shape)
print(y_class,'hthth---------------------')

import pickle
fids,f_w2v1,f_w2v1_test=pickle.load(open('../data/wordvec/f_w2v_tfidf.100.cache','rb'))  #训练集与测试集的w2v与tfidf
fids1,f_w2v2,f_w2v2_test=pickle.load(open('../data/wordvec/f_word_svd.100.cache','rb'))  #降维文本tfidf
print(f_w2v1.shape,'jhh')
print(f_w2v1_test.shape,'iuiu')

print(f_w2v2.shape,'dsdsd')
print(f_w2v2_test.shape,'ytyt')

# 导入id，tfidf_w2v  ,降维的tfidf

# f_w2v=np.concatenate((f_w2v1,f_w2v2),axis=0)
f_w2v=np.concatenate((f_w2v1,f_w2v2),axis=1)
# f_w2v_test=f_w2v1_test
f_w2v_test = np.concatenate((f_w2v1_test,f_w2v2_test),axis=1)
print(f_w2v.shape)

#--------------- MCNN ---------------
class MCNN(object):
    '''
    使用word2vec*tfidf的cnn并与人工特征混合，接口与sklearn分类器一致
    '''
    def __init__(self,cnn_input_dim,num_class=4):
        self.num_class=num_class
        self.build(cnn_input_dim)
        
    
    def build(self,vector_dim):
        #句子特征
        model=Sequential()
        # model.add(Convolution2D(100,1,vector_dim,input_shape=(2,100,vector_dim),activation='relu'))
        model.add(Conv2D(256, kernel_size=(5, vector_dim), activation='relu', input_shape=(100,vector_dim,1)))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(5,1)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(50,activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(4,activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'],)
        
        self.model=model
        self.earlyStopping=EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')  #停止条件
        # self.checkpoint=ModelCheckpointPlus(filepath='weights.hdf5',monitor='val_loss',verbose_show=20)  #生成权重文件
        
    def fit(self,X,y,Xvi=None,yvi=None):
        yc=to_categorical(y)
        if Xvi is None:
            self.model.fit(X,yc,nb_epoch=1000,verbose=0,validation_split=0.2,batch_size=32,callbacks=self.earlyStopping)
        else:
            ycvi=to_categorical(yvi)
            self.model.fit(X,yc,nb_epoch=1000,verbose=0,validation_data=[Xvi,ycvi],
                           batch_size=10)
        # self.model.load_weights('weights.hdf5')
        return self.model
    
    def predict(self,X):
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)
    
#-----------------XGBoost --------------------------------

import xgboost as xgb
class XGB(object):
    def __init__(self):
        self.params = {
            'booster': 'gblinear',
            'eta': 0.03,
            'alpha': 0.1,
            'lambda': 0,
            'subsample': 1,
            'colsample_bytree': 1,
            'num_class': 4,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'silent': 1
        }
        pass
    
    def fit(self,X,y,Xvi=None,yvi=None):
        if Xvi is None:
            ti,vi=list(StratifiedShuffleSplit(y,test_size=0.2,random_state=100,n_iter=1))[0]
            dtrain=xgb.DMatrix(X[ti],label=y[ti])
            dvalid=xgb.DMatrix(X[vi],label=y[vi])
        else:
            dtrain=xgb.DMatrix(X,label=y)
            dvalid=xgb.DMatrix(Xvi,label=yvi)
        watchlist=[(dtrain,'train1'),(dvalid,'val')]
        self.model=xgb.train(self.params,dtrain,early_stopping_rounds=200,evals=watchlist,verbose_eval=100)
        return self.model

    def predict(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        print(X.shape,'shapes')
        return self.model.predict(xgb.DMatrix(X))

#----------- 获取特征 ------------
'''
传入feature集合整合成特征集
'''
def get_xs(fs):
    if len(fs) == 1:
        return fs[0]

    if type(fs[0]) == csr_matrix:
        return sparse.hstack((fs)).tocsc()
    else:
        tmp = []
        for f in fs:
            if type(f) == csr_matrix:
                tmp.append(f.toarray())
            else:
                tmp.append(f)
        return np.hstack(tmp)

def get_xgb_X(f_train):
    f_text,f_tongji = f_train
    f_tongji = f_tongji.astype(float)
    print(f_tongji,'uyuyaaa')
    fs=[f_text[1],f_tongji] #字特征tfidf

    print(fs[0].shape, 'X')
    print(fs[1].shape, 'X')
    X=get_xs(fs)
    print(X.shape,'X')
    return X

#--------------------mcnn2-------------------------------------
def get_mcnn2_X(xtype='train'):
    if xtype=='train':
        f_text,f_tongji=f_train
        f_tongji = f_tongji.astype(float)
        x_cnn=f_w2v
    else:
        f_text,f_tongji=f_test
        f_tongji = f_tongji.astype(float)
        x_cnn=f_w2v_test

    fs=[f_text[5],f_text[3],f_tongji]  #字1维 tfidf  降维后的
    
    if xtype=='train':
        fs+=[em_xgb.get_next_input()[:146339]]  #获取每个
        print(fs,'trtr')
    else:
        fs+=[em_xgb.get_next_input()[146339:]]
    
    x_ext=get_xs(fs).toarray()
    return [x_cnn,x_ext]

if __name__=='__main__':
    filename='../data/feature/ping.class.feature'    #神经网络模型
    print('将输出文件：',filename)
    if ReTrain==True or os.path.exists(filename)==False:
        X=f_w2v
        X_test=f_w2v_test
        np.random.seed(100)
        #
        # #----mcnn model-----
        em_mcnn=StackEnsemble(lambda:MCNN(100),multi_input=False)#Lambda用来编写不需要训练的层
        f2_cnn=em_mcnn.fit(X,y_class)
        print(f2_cnn,'jhjhjh')#输出模型
        f2_cnn_test=em_mcnn.predict(X_test)
           
        #----xgb model-------
        X=get_xgb_X(f_train)  #文本与数字特征
        X_test=get_xgb_X(f_test)
        em_xgb=StackEnsemble(lambda:XGB())  #传入模型结构
        print(y_class,'uyuyuoioi')
        # print(X,'iyiyiyiy')
        f2_xgb=em_xgb.fit(X,y_class)
        f2_xgb_test=em_xgb.predict(X_test)
        print(f2_xgb_test.shape,'ytytyt')
        
        #----mcnn2 model-----
        X=get_mcnn2_X('train')
        print(X[0].shape,'X[0]')
        print(X[1].shape,'X[1]')
        X_test=get_mcnn2_X('test')
        np.random.seed(100)

        em_mcnn2=StackEnsemble(lambda:MCNN2(X[0].shape[2],X[1].shape[1],num_channel=1),multi_input=True)
        f2_cnn=em_mcnn2.fit(X,y_class)
        f2_cnn_test=em_mcnn2.predict(X_test)

        #-----------------------特征输出-------------
        import pickle
        pickle.dump([em_xgb.get_next_input(),em_mcnn.get_next_input(),em_mcnn2.get_next_input()],open(filename,'wb'))
        # pickle.dump([em_xgb.get_next_input()],open(filename,'wb'))
    print('程序运行完毕')
