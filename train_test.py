import codecs
import pandas as pd
import pickle

def get_pickle(inpuths):
	merge_file = inpuths['merge_data']
	train_file = open(inpuths['csv_train_file'],encoding='utf-8').read()
	valid_file = open(inpuths['csv_valid_file'],encoding='utf-8').read()
	with open(merge_file, 'a',encoding='utf-8') as f:
		f.write(train_file)
		f.write(valid_file)

	merge_pickle = inpuths['merge_pickle']
	data = pd.read_csv(merge_file,names=['id', 'content','label'])
	with codecs.open(merge_pickle, 'wb')as f:  # 输出带表头的pickle内容文件
		pickle.dump(data, f)


