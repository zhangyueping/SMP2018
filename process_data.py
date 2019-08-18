import numpy as np
import pandas as pd
import codecs
from sklearn.preprocessing import LabelEncoder
import pickle
from gensim.models.word2vec import Word2Vec
import os
train_file = r'../data/train data/train_data.txt'
train_labels=r'../data/train data/train_label.txt'
test_file = r'../data/test data/test_data.txt'
test_label = r'../data/test data/test_label.txt'

merge_file = r'../data/preprocess/merge_data.txt'
feature_file = r'../data/feature/feature.txt'
concact_all_fea_con = r'../data/feature/all_fea_con_label.txt'

features_label = r'../data/feature/features.v1.pkl'
features_label2 = r'../data/feature/features.v2.pkl'



def all_concact(feature_file,merge_file,concact_all_fea_con):
    '''
    将feature文件，train,test与text文件合并
    :return:
    '''
    feature = pd.read_csv(feature_file,names = ['id','sen','sum_words','averge_words','sum_maohao','averge_maohao','count_num','averge_num','per_word_len','noun','adv','verb','adj','pron'])
    all = pd.read_csv(merge_file,names = ['id','content','label'])
    labels = LabelEncoder()  # 标准化
    labels.fit(all.label)
    all['label'] = labels.transform(all.label)
    data = pd.merge(feature, all, on=['id'], how='left')
    data.to_csv(concact_all_fea_con, encoding='utf-8',header=False, index=False)



def get(train_data,i):
    return [item[i] for item in train_data]

def get_train_test_data(train_labels,concact_all_fea_con):
    '''
    将数字特征，文本内容，id，label放在列表里，取着方便
    '''
    train_file  = codecs.open(train_labels,encoding='utf-8')
    train_len = len(train_file.readlines())
    print(train_len,'oioioi')
    train_data = []
    test_data = []
    file = codecs.open(concact_all_fea_con,encoding='utf-8')
    lines = file.readlines()
    for line in lines[:train_len]:
        lists = line.strip().split(',')
        id = lists[0]
        sentence_len = lists[1]
        sum_words = lists[2]
        averge_words = lists[3]
        sum_maohao = lists[4]
        averge_maohao = lists[5]
        count_num = lists[6]
        averge_num = lists[7]
        per_word_len = lists[8]
        noun = lists[9]
        adv = lists[10]
        verb = lists[11]
        adj = lists[12]
        pron = lists[13]
        content = lists[14]
        label = lists[15]
        train_data.append([id,sentence_len,sum_words,averge_words,sum_maohao,averge_maohao,count_num,averge_num,per_word_len,noun,adv,verb,adj,pron,content,label])
    for line in lines[train_len:]:
        lists = line.strip().split(',')
        id = lists[0]
        sentence_len = lists[1]
        sum_words = lists[2]
        averge_words = lists[3]
        sum_maohao = lists[4]
        averge_maohao = lists[5]
        count_num = lists[6]
        averge_num = lists[7]
        per_word_len = lists[8]
        noun = lists[9]
        adv = lists[10]
        verb = lists[11]
        adj = lists[12]
        pron = lists[13]
        content = lists[14]
        label = lists[15]
        test_data.append([id, sentence_len, sum_words, averge_words, sum_maohao, averge_maohao, count_num, averge_num, per_word_len,
             noun, adv, verb, adj, pron, content, label])

    return train_data,test_data

if os.path.isfile(concact_all_fea_con):
    print('已生成结合文件')
else:
    print('将数字，Id，content结合 train_test')
    all_concact(feature_file,merge_file,concact_all_fea_con)
train_data,test_data=get_train_test_data(train_labels,concact_all_fea_con)
print('train_data:%d'%(len(train_data)))
print('test_data:%d'%(len(test_data)))
#其实也就是将特征与id，label放一块，费这么大劲，已经生成了特征文件了啊

if os.path.isfile(features_label):
    print('已生成feature.v1文件')
else:
    pickle.dump([train_data,test_data],open(features_label,'wb'))  #将数字特征，id ,content,label全部写进feature
    print('pickle文件生成')



from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
'''获得计数类文本特征'''
# def get_f_cnter(xs,min_df=5):
#     cnter=CountVectorizer(min_df=min_df)
#     cnter.fit(xs)
#     f_xs=[cnter.transform(x) for x in xs]
#     print(f_xs)
#     return f_xs

'''将整数转换占比例形式'''
# def to_rate(a):
#     b=np.sum(a,axis=1)
#     return a/np.outer(b,np.ones(a.shape[1]))

#-------------------1. 文本特征-------------------------------------
if os.path.isfile(features_label2):
    print('已生成feature.v2文件')
else:
    print('正在生成feature.v2文件')
    '''
    tfidf类特征 
    '''
    from sklearn.decomposition import TruncatedSVD

    content = get(train_data, 14), get(test_data, 14)  # 得到信息
    xs = content

    '''词tfidf特征 1gram'''
    tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 1))  # min_df表示单词出现的最少document个数  2-gram
    # ngram_range要提取的n-gram的n-values的下限和上限范围。
    a = tfidf.fit(xs[0])  # 训练集
    f_word = [tfidf.transform(x) for x in xs]
    print('f_word', f_word[0].shape)
    print('f_word', f_word[1].shape, 'oioioi')

    '''字tfidf特征 1gram'''
    tfidf2 = TfidfVectorizer(min_df=2, ngram_range=(1, 1), analyzer='char')  # 2-graml
    tfidf2.fit(xs[0])
    f_letter = [tfidf2.transform(x) for x in xs]
    # print('f_letter',f_letter[0].shape)

    '''svd降维 1'''
    svd = TruncatedSVD(n_components=100)
    svd.fit(f_word[0])  # 将内容的tf-idf值进行降维
    f_word_pca = svd.transform(f_word[0]), svd.transform(f_word[1])
    print('f_word_pca', f_word_pca[0].shape)

    svd = TruncatedSVD(n_components=100)
    svd.fit(f_letter[0])  # 对字的tf-idf降维300维
    f_letter_pca = svd.transform(f_letter[0]), svd.transform(f_letter[1])

    '''词tfidf特征 2 gram1'''
    tfidf3 = TfidfVectorizer(min_df=3, ngram_range=(1, 2))
    tfidf3.fit(xs[0])
    f_word_n1 = [tfidf3.transform(x) for x in xs]
    print('f_word_n1', f_word_n1[0].shape)

    '''字tfidf特征 2gram1'''
    tfidf4 = TfidfVectorizer(min_df=3, ngram_range=(1, 2), analyzer='char')  # ngram_range词组切分的长度范围
    tfidf4.fit(xs[0])
    f_letter_n1 = [tfidf4.transform(x) for x in xs]
    print('f_letter_n1', f_letter_n1[0].shape)

    f_text = [f_word, f_letter, f_word_pca, f_letter_pca, f_word_n1, f_letter_n1]

    # -------------------2. 统计特征-------------------------------------------------------


    '''统计类特征'''


    def get_fea(data):
        fs = []
        for line in data:
            sentence_len = line[1]
            sum_words = line[2]
            averge_words = line[3]
            sum_maohao = line[4]
            averge_maohao = line[5]
            count_num = line[6]
            averge_num = line[7]
            per_word_len = line[8]
            noun = line[9]
            adv = line[10]
            verb = line[11]
            adj = line[12]
            pron = line[13]
            fs.append([sentence_len, sum_words, averge_words, sum_maohao, averge_maohao, count_num, averge_num, per_word_len,
                 noun, adv, verb, adj, pron])

        return np.array(fs)


    f_tongji = get_fea(train_data), get_fea(test_data)

    # ----------------------5. 抽取y值--------------------------------------
    '''抽取y值'''
    ids = get(train_data, 0), get(test_data, 0)
    y_label = get(train_data, 15)
    ys = np.array(y_label)
    print(ys, 'ysysysys')
    f_text_train = [item[0] for item in f_text]
    f_text_test = [item[1] for item in f_text]

    f_train = [f_text_train, f_tongji[0]]
    f_test = [f_text_test, f_tongji[1]]

    f_content = get(train_data, 14), get(test_data, 14)

    pickle.dump([ids, ys, f_train, f_test, f_content], open(features_label2, 'wb'))  # 将特征全部转变为numpy array保存
    print('features.v2.pkl输出完毕!')
    print('...')


#-----------------------6. 输出word2vec---------------------------------------

def load_w2v(dim=100):#载入词向量
    if dim==100:
        return Word2Vec.load('../data/wordvec/model100_20180703')
    return None
'''
传入单词表，返回词向量集合
'''
def get_word_vectors(words,w2v_model=None):
    if w2v_model==None:
        w2v_model=load_w2v()
    vectors=[]
    cnt=0
    for w in words:
        if w in w2v_model:
            vectors.append(w2v_model[w])
        else:
            vectors.append(np.zeros(w2v_model.vector_size))
            cnt+=1
    print('不在词表中的词数量：',cnt)
    print(len(vectors),'lengeh_vector')
    return np.array(vectors)

'''
切分句子
'''
def split_sens(contents):
    sens=[]
    ids=[]
    count = 0
    for item in contents:
        c_sens=item.split('。')
        count += len(c_sens)
        # print(len(c_sens),'-------------------')
        ids.append(range(len(sens),len(sens)+len(c_sens)))
        sens.extend(c_sens)
    print(count)
    return sens,ids

'''
分配句子
'''
def get_f_w2v(xs,ids,vector_size=100):
    x_cnn=[]
    max_dim=50
    for indexs in ids:
        item=[]
        for i in indexs[:min(max_dim,len(indexs))]:
            item.append(xs[i])
        if len(item)<max_dim:
            for i in range(max_dim-len(item)):
                item.append(np.zeros(vector_size))
        #random.shuffle(item)
        x_cnn.append(np.array(item))
    return np.array(x_cnn)


IsRefresh = False
import os
import pickle
f_content = get(train_data, 14), get(test_data, 14)
ids = get(train_data, 0), get(test_data, 0)
filename = '../data/wordvec/f_w2v_tfidf.100.cache'
if os.path.exists(filename) == False or IsRefresh:
    '''split to sentences训练集'''
    sens, tids = split_sens(f_content[0])

    tfidf = TfidfVectorizer(min_df=2)
    f_tfidf_raw = tfidf.fit_transform(sens)
    print('tfidf dim:', f_tfidf_raw.shape)

    '''get word2vec library'''
    model = load_w2v(100)
    vectors = get_word_vectors(tfidf.get_feature_names(), model)
    f_flatten_w2v = f_tfidf_raw * vectors
    print(f_flatten_w2v.shape, 'f_flatten_w2v.shape')

    f_w2v = get_f_w2v(f_flatten_w2v, tids, model.vector_size)
    print('f_w2v dim:', f_w2v.shape)

    f_w2v_train = f_w2v.reshape(f_w2v.shape[0], f_w2v.shape[1], f_w2v.shape[2], 1)
    print('f_w2v dim:', f_w2v_train.shape)
    test_sens, test_ids = split_sens(f_content[1])

    f_tfidf_raw_test = tfidf.transform(test_sens)

    f_flatten_w2v_test = f_tfidf_raw_test * vectors

    f_w2v_test = get_f_w2v(f_flatten_w2v_test, test_ids, model.vector_size)
    f_w2v_tests = f_w2v_test.reshape(f_w2v_test.shape[0], f_w2v_test.shape[1], f_w2v_test.shape[2], 1)

    pickle.dump([ids, f_w2v_train, f_w2v_tests], open(filename, 'wb'))

# else:
#     fids, f_w2v, f_w2v_test = pickle.load(open(filename, 'rb'))

print(filename, '输出完毕')
print('...')

# --------------------7. 输出降维的tfidf特征---------------------------
'''word svd300'''


def get_f_cnn(xs, ids, vector_size=100):
    x_cnn = []
    max_dim = 50
    for indexs in ids:
        item = []
        for i in indexs[:min(max_dim, len(indexs))]:
            item.append(xs[i])
        if len(item) < max_dim:
            for i in range(max_dim - len(item)):
                item.append(np.zeros(vector_size))
        # random.shuffle(item)
        x_cnn.append(np.array(item))
    print(len(x_cnn))
    return np.array(x_cnn)


filename = '../data/wordvec/f_word_svd.100.cache'

sens, tids = split_sens(f_content[0] + f_content[1])

tfidf = TfidfVectorizer(min_df=2)
f_tfidf = tfidf.fit_transform(sens)
print('tfidf dim:', f_tfidf.shape)

svd = TruncatedSVD(n_components=100)
f_svd = svd.fit_transform(f_tfidf)
print(f_svd.shape,'jhjhjh')
f_cnn = get_f_cnn(f_svd, tids)
f_cnns = f_cnn.reshape(f_cnn.shape[0],f_cnn.shape[1],f_cnn.shape[2],1)
print(f_cnns.shape,'f_cnn')
pickle.dump([ids, f_cnns[:146339],f_cnns[146339:]], open(filename, 'wb'))  #写入文件tfidf

print(filename, '输出完毕')
print('...')

# --------------------------8. 输出字的tfidf降维特征------------------------
'''letter svd300'''
# filename =  '../data/wordvec/f_letter_svd.300.cache'
#
# sens, tids = split_sens(f_content[0] + f_content[1])
#
# tfidf = TfidfVectorizer(min_df=3, analyzer='char')
# f_tfidf = tfidf.fit_transform(sens)
# print('tfidf dim:', f_tfidf.shape)
#
# svd = TruncatedSVD(n_components=300)
# f_svd = svd.fit_transform(f_tfidf)
# print(f_svd)
# f_cnn = get_f_cnn(f_svd, tids)
# pickle.dump([ids, f_cnn[:117134],f_cnn[117134:]], open(filename, 'wb'))
#
# print(filename, '输出完毕')
# print('...')
