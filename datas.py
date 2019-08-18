#是否修改了吗
import pandas as pd
import codecs
import jieba
class Flush(object):
	def __init__(self,inpuths):
		self.input_file_train = inpuths['dic_train_file']
		self.stop_words = inpuths['stop_words']
		self.csv_file_train = inpuths['csv_train_file']

		self.input_file_valid = inpuths['dic_valid_file']
		self.csv_file_valid = inpuths['csv_valid_file']

	def get_content(self,acc):
		if acc == 'train':
			content = self.dic_row['内容']
			id = self.dic_row['id']
			label = self.dic_row['标签']
			return id,content,label

		if acc == 'valid':
			content = self.dic_row['内容']
			id = self.dic_row['id']
			return id,content

	def stopwordlist(self):
		stopwords = [line.strip() for line in open(self.stop_words,'r',encoding='utf-8').readlines()]
		return stopwords

	def seg_sentence(self,content):
		content_seg = jieba.cut(content)
		stop_words = self.stopwordlist()
		outstr = ''
		for word in content_seg:
			if word not in stop_words:
				outstr += word
				outstr += ' '
		return outstr

	def merge_train(self):#生成带分词的csv文件
		write_file = codecs.open(self.csv_file_train,'w+','utf-8',errors='ignore')
		with codecs.open(self.input_file_train,'r','utf-8')as file:
			for row in file:
				self.dic_row = eval(row)
				id,content,label=self.get_content('train')
				content_seg = self.seg_sentence(content.strip())
				strs = str(id) + ',' + content_seg +','+ label+'\n'
				write_file.write(strs)


	def merge_valid(self):#生成带分词的csv文件
		write_file = codecs.open(self.csv_file_valid,'w+','utf-8',errors='ignore')
		with codecs.open(self.input_file_valid,'r','utf-8')as file:
			for row in file:
				self.dic_row = eval(row)
				id,content=self.get_content('valid')
				content_seg = self.seg_sentence(content.strip())
				strs = str(id) + ',' + content_seg +','+ '\n'
				write_file.write(strs)













