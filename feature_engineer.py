import codecs
import re
import pickle
import pandas as pd
import jieba.posseg as pseg
class Feature(object):
	def __init__(self,inpaths,outpaths):#输入内容，标签
		self.all_file = inpaths['merge_data']
		self.features_file = outpaths['feature']
		self.feature_pickle = outpaths['feature_pickle']


	def build_feature(self):
		#提取数据特征
		file = codecs.open(self.features_file, 'a+',encoding='utf-8')
		file2 = codecs.open(self.all_file,encoding='utf-8')
		for i in file2:
			lists = i.strip().split(',')
			contents = lists[1]
			sentences = contents.strip().split('。')
			sentence_len = len(sentences)
			words_len = []
			maohao_len = []
			nums_len = []
			word_len = []
			flag_list = []
			nums_re = r'\d{0,10}\.{0,1}\d{0,20}'
			for sentence in sentences:
				words_per = sentence.strip().split(' ')
				word_len_ = []
				for word in words_per:
					word_len_.append(len(word))
				average_len_word = sum(word_len_)/len(word_len_)
				word_len.append(average_len_word)
				words_len.append(len(words_per))
				maohao_num = words_per.count('：')
				maohao_len.append(maohao_num)
				nums_list = re.findall(nums_re,sentence)
				your_list = [x for x in nums_list if x != '']
				nums_len.append(len(your_list))
				words = pseg.cut(sentence)
				flag = []
				for key in words:
					flag.append(key.flag)
				flag_list.extend(flag)
			noun = self.check_pos_tag(flag_list,'noun')
			adv = self.check_pos_tag(flag_list,'adv')
			verb = self.check_pos_tag(flag_list,'verb')
			adj = self.check_pos_tag(flag_list,'adj')
			pron = self.check_pos_tag(flag_list,'pron')
			per_word_len =  sum(word_len) / len(sentences)
			sum_words,averge_words = [sum(words_len),sum(words_len)/len(sentences)]  #平均每句话的单词个数
			sum_maohao,averge_maohao = [sum(maohao_len),sum(maohao_len)/len(sentences)] #平均每句话的冒号个数
			count_num,averge_num = [sum(nums_len),sum(nums_len)/len(nums_len) if len(nums_len) != 0 else 0] #平均每句话的数字个数
			strs = lists[0]+','+str(sentence_len)+','+str(sum_words)+','+str(averge_words)+','+str(sum_maohao)+','+str(averge_maohao)+','+str(count_num)+','+str(averge_num)+','+str(per_word_len)+','+str(noun)+','+str(adv)+','+str(verb)+','+str(adj)+','+str(pron)
			file.write(strs+'\n')

	def check_pos_tag(self,flag_list,pos):
		pos_family = {
		'noun':['n','nr','ns','nt','nz'],
		'adv':['d'],
		'verb':['v','vd','vn'],
		'adj':['a','ad','an'],
		'pron':['p']
		}
		cnt = 0
		try:
			for tup in flag_list:
				if tup in pos_family[pos]:
					cnt += 1
		except:
			pass
		return cnt

	def get_feature_pickle(self):
		'''
		将feature文件存为带表头的pickle文件
		'''
		self.build_feature()
		data = pd.read_csv(self.features_file,names=['id', 'sen', 'sum_words', 'averge_words', 'sum_maohao', 'averge_maohao', 'count_num', 'averge_num','per_word_len','noun','adv','verb',
		'adj','pron'])
		with codecs.open(self.feature_pickle, 'wb')as f:  # 输出带表头的pickle内容文件
			pickle.dump(data, f)



