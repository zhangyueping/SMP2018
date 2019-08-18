import pandas as pd

# id = [i for i in range(20,40)]
# pre = [j for j in range(0,20)]
# result = pd.DataFrame(pre, columns=['label'])
#
# result.insert(0,'id',id)
#
# result.to_csv('a.csv', index=None)
# print('result saved!')


# file = open('../data/output/temp.csv',encoding='utf-8')
# csv_file_id = open('../data/valid data/resolution_valid.txt',encoding='utf-8')
# temp = 'temp.csv'
# ids = []
# for line in csv_file_id:
# 	id = line.strip().split(',')[0]
# 	print(id)
# 	ids.append(id)
# print(len(ids))
#
# result = pd.read_csv(file,names=['label'] )
# print(result)
# print(len(result))
# result.insert(0,'id',ids)
#
# result.to_csv(temp, index=None,header = None)
# print('result saved!')

import codecs
# def get_valid():
#     f2 = codecs.open('../data/valid data/write_valid_utf1.txt', 'w', 'utf-8')
#     # 处理文件  变为中文
#     count = 0
#     count1 = 0
#     with codecs.open('../data/valid data/validation.txt', 'r')as file:
#         for i in file:
#             # 变成中文
#             try:
#                 a = i.strip().encode('utf-8').decode('unicode-escape')
#                 print(a)
#                 f2.write(a + '\n')
#                 count1 += 1
#             except Exception as e:
#                 print(e, i, 'gtgtgt')
#                 count += 1
#
#         print(count,count1)
# get_valid()
    # file = codecs.open('../data/valid data/write_valid.txt', 'w', encoding='utf-8')
    # with open('../data/valid data/write_valid_utf.txt', encoding='utf-8') as finp:
    # 	count = 0
    # 	count2 = 0
    # 	pattern1 = r'\"内容\": \"(.*)\"}'
    # 	for line in finp.readlines():
    # 		try:
    # 			line_final = line[:22] + re.sub('\"|\t|\r|\\\\|\\\\n', '', re.findall(pattern1, line)[0]) + '"}'
    # 			print(line_final)
    # 			count2 += 1
    # 			# pickle.dump(line_final+'\n',file)
    # 			file.write(line_final + '\n')
    # 		# a = eval(line)
    # 		# print(a)
    # 		except Exception as e:
    # 			print(e, 'gfrgfdgfdg', line)
    # 			count += 1
    # 	print(count, count2)
# import re
# file = codecs.open('../data/valid data/write_valid.txt', 'w', encoding='utf-8')
# with open('../data/valid data/write_valid_utf.txt', encoding='utf-8') as finp:
#     count = 0
#     count2 = 0
#     pattern1 = r'\"内容\": \"(.*)\"}'
#     for line in finp.readlines():
#         try:
#             line_final = line[:22] + re.sub('\"|\t|\r|\\\\|\\\\n', '', re.findall(pattern1, line)[0])+'"}'
#             # print(line_final)
#             count2 += 1
#             # pickle.dump(line_final+'\n',file)
#             file.write(line_final + '\n')
#             # a = eval(line)
#             # print(a)
#         except Exception as e:
#             print(e, 'gfrgfdgfdg', line)
#             count += 1
#     print(count, count2)

# import re
# def get_train():
# 	# f2 = codecs.open('../data/train data/write_train_utf.txt', 'a+', 'utf-8')
# 	# # 处理文件  变为中文
# 	# with codecs.open('../data/train data/training.txt', 'r')as file:
# 	# 	for i in file:
# 	# 		# 变成中文
# 	# 		a = i.strip().encode('utf-8').decode('unicode-escape')
# 	# 		f2.write(a + '\n')
#
# 	file = codecs.open('../data/train data/write_train.txt', 'a+', encoding='utf-8')
# 	with open('../data/train data/write_train_utf.txt', encoding='utf-8') as finp:
# 		count = 0
# 		count2 = 0
# 		pattern1 = r'\"内容\": \"(.*)\", \"id\"'
#
# 		for line in finp.readlines():
# 			try:
# 				line_final = line[:22] + re.sub('\"|\t|\r|\\\\|\\\\n', '', re.findall(pattern1, line)[0]) + '\", ' + \
# 				             re.findall(r'\"id\":.*', line)[0]
# 				count2 += 1
# 				# print(line_final)
# 				# pickle.dump(line_final+'\n',file)
# 				file.write(line_final + '\n')
# 			except Exception as e:
# 				print(e, 'gfrgfdgfdg', line)
# 				count += 1
# 		print(count, count2)
#
# get_train()



file = open('../data/train data/resolution_train_new.txt',encoding='utf-8')
length = len(file.readlines())
print(length)