#coding:utf-8
import codecs
import re
'''
预处理文本
1、先在源文件中training手动去除\n,\t,\r等
2、由unicode转化为中文，写入文件write_utf.txt
3、去除多余引号，使其成为字典格式，写入write2.txt
若有两个文件，则分别生成 训练集与测试集
'''

class Text_process(object):
    def get_train(self):
        f2 = codecs.open('../data/train data/write_train_utf.txt','a+','utf-8')
        #处理文件  变为中文
        with codecs.open('../data/train data/training.txt','r')as file:
            for i in file:
                # 变成中文
                a = i.strip().encode('utf-8').decode('unicode-escape')
                f2.write(a+'\n')


        file = codecs.open('../data/train data/write_train.txt','a+',encoding='utf-8')
        with open('../data/train data/write_train_utf.txt',encoding='utf-8') as finp:
            count = 0
            count2 = 0
            pattern1 = r'\"内容\": \"(.*)\", \"id\"'
            for line in finp.readlines():
                try:
                    line_final = line[:22]+re.sub('\"|\t|\r|\\\\|\\\\n','',re.findall(pattern1,line)[0])+'\", '+re.findall(r'\"id\":.*',line)[0]
                    count2 += 1
                    # pickle.dump(line_final+'\n',file)
                    file.write(line_final+'\n')
                except Exception as e:
                    print(e,'gfrgfdgfdg',line)
                    count += 1
            print(count,count2)

    def get_valid(self):
        f2 = codecs.open('../data/valid data/write_valid_utf.txt', 'w', 'utf-8')
        # 处理文件  变为中文
        count = 0
        with codecs.open('../data/valid data/validation.txt', 'r')as file:
            for i in file:
                    # 变成中文
                try:
                    a = i.strip().encode('utf-8').decode('unicode-escape')
                    f2.write(a + '\n')
                except Exception as e:
                    print(e,i,'gtgtgt')
                    count += 1

            print(count)

        file = codecs.open('../data/valid data/write_valid.txt', 'w', encoding='utf-8')
        with open('../data/valid data/write_valid_utf.txt', encoding='utf-8') as finp:
            count = 0
            count2 = 0
            pattern1 = r'\"内容\": \"(.*)\"}'
            for line in finp.readlines():
                try:
                    line_final = line[:22] + re.sub('\"|\t|\r|\\\\|\\\\n', '', re.findall(pattern1, line)[0])+'"}'
                    print(line_final)
                    count2 += 1
                    # pickle.dump(line_final+'\n',file)
                    file.write(line_final + '\n')
                    # a = eval(line)
                    # print(a)
                except Exception as e:
                    print(e, 'gfrgfdgfdg', line)
                    count += 1
            print(count, count2)





