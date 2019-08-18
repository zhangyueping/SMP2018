import os
from 评测.src.get_label import get_train_label
from 评测.src.datas import Flush
from 评测.src.model import Model
from 评测.src.feature_engineer import Feature
from 评测.src.train_test import get_pickle
from 评测.src.text_process import Text_process
inpaths = {
    'stop_words': '../data/preprocess/stopwords.txt',#停用词

    'dic_train_file': '../data/train data/write_train_new.txt',#标准字典文件
    'csv_train_file':'../data/train data/resolution_train_new.txt',#带分词的表格文件
    'train_label':'../data/train data/train_label_new.txt',

    'dic_valid_file': '../data/valid data/write_valid_new.txt',  # 标准字典文件
    'csv_valid_file': '../data/valid data/resolution_valid_new.txt',  # 带分词的表格文件

    'merge_data':'../data/preprocess/merge_data_new.txt',
    'merge_pickle':'../data/merge_pickle_new.txt'


}

outpaths = {
    'feature':'../data/feature/feature_new.txt',#特征文件不带表头
    'feature_pickle':'../data/feature/feature_pickle_new.txt',#特征文件pickle带表头

    'stack_file': '../data/output/stack_new.txt',#存放文本特征文件tf-idf

    'result_label': '../data/output/temp_new.csv',  # 存放分类结果

    'outfeatures':'../data/feature/ping.class.feature',


}






def predict():#需要特征文件
    '''
    文本特征tf-idf值与特征结合
    :return:
    '''
    print('################## Label  #######################')
    model_label = Model(inpaths['merge_pickle'],outpaths['stack_file'],outpaths['feature_pickle'],inpaths["train_label"])
    if os.path.isfile(outpaths['stack_file']):
        print('stacked~ 开始训练第二级模型')
    else:
        print('stacking')
        model_label.stacking()  #先生成stack文件
    model_label.concat_features()
    model_label.fit_transform(outpaths['result_label'])






if __name__ == '__main__':
    '''
    预处理训练集与测试集文件
    '''
    if os.path.isfile(inpaths['dic_train_file']):
         print('已处理')
    else:
         print('正在处理。。。')
         text = Text_process()
         text.get_train()
         text.get_valid()

    '''
    生成分词文件
    '''
    if os.path.isfile(inpaths['csv_train_file']):
         print('已分词')
    else:
        print('正在分词。。。')
        csv = Flush(inpaths)
        csv.merge_train()  #生成内容文件，供modle模块使用
        csv.merge_valid()


    '''
   将训练集与测试集合并
    '''
    if os.path.isfile(inpaths['merge_data']):
        print('已合并')
    else:
        print('切分数据。。。')
        get_pickle(inpaths)



    '''
    生成label文件
    '''
    if os.path.isfile(inpaths['train_label']):
        print('已生成标签文件')
    else:
        print('正在生成标签。。。')
        get_train_label(inpaths)



    '''
    得到特征文件
    '''
    if os.path.isfile(outpaths['feature_pickle']):
        print('已生成数字特征')
    else:
        print('正在生成数字特征。。。')
        feature = Feature(inpaths, outpaths)
        feature.get_feature_pickle()

    if os.path.isfile(outpaths['result_label']):
        print('已生成结果文件')
    else:
        print('正在预测。。。')
        predict()







