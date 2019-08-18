import codecs


def get_train_label(inpaths):
    '''
    读取表格文件
    :return: 储存标签文件
    '''
    f = codecs.open(inpaths['csv_train_file'], 'r', 'utf-8')
    file = codecs.open(inpaths['train_label'], 'w', 'utf-8')
    for i in f:
        s = i.strip().split(',')
        # print(s)
        a = s[0] + ',' + s[-1] + '\n'
        file.write(a)



