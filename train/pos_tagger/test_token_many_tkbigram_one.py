import tensorflow as tf
import numpy as np
import pickle
from tokenization_kobert import KoBertTokenizer

import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
model = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger.model',compile=False)
with open('kcc150_all_tag_dict.pkl','rb') as f:
    tag_dict = pickle.load(f)
    tag_dict = { v:k for k,v in tag_dict.items() }
    #print(tag_dict.keys())
#with open('eum_vocab.pkl','rb') as f:
#    vocab = pickle.load(f)
max_len = 200
import time
start = time.time()
# print(vocab.keys())
# x = '안녕하세요 국민대학▁김정민이다▁.'
# x = '내▁눈을▁본다면▁밤하늘의▁별이▁되는▁기분을▁느낄▁수▁있을▁거야'
# x = '나는▁밥을▁먹고▁집에▁갔다.'

def make_input(x_t):
	masks = []
	segments = []

	for x in x_t:
		# for d in x:
		d_ = [t for t in x if t != 1]
		# print(len(d_))
		mask = [1]*len(d_) + [0] * (max_len-len(d_))
		masks.append(mask)
		segment = [0] * max_len
		segments.append(segment)
		# masks = masks * 100
		# segments = segments * 100
	masks = np.array(masks)
	segments = np.array(segments)

	return masks,segments

def w2i(x_t):
	res = []
	for i in x_t:
		x = i.replace('  ',' ').replace('+',' ')
		#x = list(x)
		#x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 
		x = tokenizer.tokenize('[CLS] '+ x+' [SEP]') 
		x = tokenizer.convert_tokens_to_ids(x)#[2]+x+[3]

		x = x + [1] * (max_len-len(x))
		res.append(x)

	x = np.array(res)

	return x

def predict(x):
	# x = x.replace('  ',' ').replace(' ','▁')
	# x = list(x)
	x_temp = x
	x_len = len(x) + 2
	x = w2i(x)
	# print(x)
	# x = [x]
	# x = x * 100
	# print(x)
	# print(x.shape)

	masks,segments = make_input(x)
	x = np.array(x)

	# print(masks.shape,segments.shape,x.shape)
	# import sys
	# sys.exit()
	# return x,x_temp,x_len
	result = model.predict([x,masks,segments])

	tagging = []
	# print(result.shape)
	for index,i in enumerate(result):
		array = []
		stop = -1
		for re_ in range(len(i)-1,0,-1):
			# print(result[re])
			tag = tf.argmax(i[re_])
			# array.append(tag)

			if tag != 3:
				stop = re_
				break
		temp = i[:stop+1]
		temp = temp[1:]
		#x_te = list(x_temp[index])
		re_tag = []
		for index_temp,te in enumerate(temp):
			tag = tf.argmax(te)
			
			re_tag.append(tag_dict[tag.numpy()])
			#if tag.numpy() == 2:
				#array.append(index_temp)
			#	x_te[index_temp] = x_te[index_temp] + '<tab>'
				# print(index,tag.numpy())
				# print(x_te)
			#elif tag.numpy() == 1:
			#    x_temp[index_temp] = x_temp[index_temp] + '▁'
		
		tagging.append(' '.join(re_tag))
		# print(tagging)
	return tagging

def predictbi(x,bi_temp):
    	# x = x.replace('  ',' ').replace(' ','▁')
	# x = list(x)
	x_temp = x
	x_len = len(x) + 2
	x = w2i(x)
	# print(x)
	# x = [x]
	# x = x * 100
	# print(x)
	# print(x.shape)

	masks,segments = make_input(x)
	x = np.array(x)
	bi_temp = np.array(bi_temp)

	# print(masks.shape,segments.shape,x.shape)
	# import sys
	# sys.exit()
	# return x,x_temp,x_len
	result = model.predict([bi_temp,x,masks,segments])

	tagging = []
	result_ = tf.argmax(result,axis=2).numpy()
	result_ = result_[:,1:]
	for rr in result_:
		re_tag = []
		for rr_index,i in enumerate(rr):
			if tag_dict[i] == '[PAD]' and rr_index < len(rr) -1 and tag_dict[rr[rr_index+1]]=='[PAD]':
				break
			re_tag.append(tag_dict[i])
		tagging.append(' '.join(re_tag))
	# print(result.shape)
	#for index,i in enumerate(result):
	#	array = []
	#	stop = -1
	#	for re_ in range(len(i)-1,0,-1):
			# print(result[re])
	#		tag = tf.argmax(i[re_])
			# array.append(tag)

	#		if tag != 3:
	#			stop = re_
	#			break
	#	temp = i[:stop+1]
	#	temp = temp[1:]
		#x_te = list(x_temp[index])
	#	re_tag = []
	#	for index_temp,te in enumerate(temp):
	#		tag = tf.argmax(te)
			
	#		re_tag.append(tag_dict[tag.numpy()])
			#if tag.numpy() == 2:
				#array.append(index_temp)
			#	x_te[index_temp] = x_te[index_temp] + '<tab>'
				# print(index,tag.numpy())
				# print(x_te)
			#elif tag.numpy() == 1:
			#    x_temp[index_temp] = x_temp[index_temp] + '▁'
		
	#	tagging.append(' '.join(re_tag))
		# print(tagging)
	return tagging
#x = x[0][:x_len]

#array = array[1:-1]
#for i in array:#range(len(x_temp)):
    #x_te = x_temp[i]
    #tag_te = array[i]

    #if i == 2:
#    x_temp[i] = x_temp[i]+'+'

# print(''.join(x_temp))
# print(array)

with open('kcc150_all_tokenbi.pkl','rb') as f:
    bigram = pickle.load(f)

result_file = open('bert_re_tkbigram_one.txt','w',encoding='utf-8')
result_file_y = open('bert_fail_tkbigram_one.txt','w',encoding='utf-8')
result_file_tk = open('bert_tk_tkbigram_one.txt','w',encoding='utf-8')
from tqdm import tqdm
import re
with open('test_data_x.txt','r',encoding='utf-8') as f:
	with open('test_data_y.txt','r',encoding='utf-8') as ff:
		count = 0
		X = []
		Y = []
		BI = []
		for _ in tqdm(range(60000)):
			l = f.readline()
			l = l.replace('\n','')
			temp_l = l
			l = re.sub(' +',' ',l)
			#l = re.sub('\.+','..',l)
			#l = re.sub('ㅋ+','ㅋㅋ',l)
			#l = re.sub('ㅎ+','ㅎㅎ',l)
			#l = re.sub('!+','!',l)
			#l = re.sub('\?+','?',l)
			#l = re.sub(',+',',',l)
			#l = re.sub('~+','~',l)
			#l = re.sub(';+',';',l)
			#l = re.sub('ㅇ+','ㅇㅇ',l)
			#l = re.sub('ㅠ+','ㅠㅠ',l)
			#l = re.sub('ㅜ+','ㅜㅜ',l)
			yy = ff.readline()
			# print(l)
			bi_npy = []
			# uni_npy = []
			l_temp = l.replace('+',' ')
			#for i in range(len(l_temp)-1):
			#	bi = l_temp[i:i+2]
				# if bi not in bigram:
				# 	bigram[bi] = biindex
				# 	biindex += 1
			#	if bi in bigram:
			#		bi_npy.append(bigram[bi])
			#	else:
			#		bi_npy.append(bigram['[UNK]'])
			#bi_npy = bi_npy + [1] * (260 - len(bi_npy))
			#bi_npy = np.array(bi_npy)
			temp = tokenizer.tokenize('[CLS] '+ l_temp +' [SEP]')
			for token_i in range(len(temp)):
				bii = temp[token_i:token_i+2]
				bii = '+'.join(bii)
				if bii in bigram:
					bi_npy.append(bigram[bii])
				else:
					bi_npy.append(bigram['[UNK]'])
			bi_npy = bi_npy + [1] * (max_len - len(bi_npy))
			bi_npy = np.array(bi_npy)
			if len(temp) <= max_len and bi_npy.shape[0] <= max_len:
				X.append(l)
				Y.append(yy)
				BI.append(bi_npy)
				count += 1
				result_file_tk.write(temp_l+'\n')
				if count % 100 ==0:
					result = predictbi(X,BI)#.replace('▁',' ')
					result_file.write('\n'.join(result)+'\n')
					result_file_y.write(''.join(Y))
					Y = []
					X = []
					BI = []
				if count == 50000:
					break
#if len(X) > 0:	
#	result = predictbi(X,BI)#.replace('▁ ',' ')
#	result_file.write('\n'.join(result)+'\n')   
#	result_file_y.write(''.join(Y))
result_file.close()
result_file_y.close()
result_file_tk.close()
end = time.time()

print(end-start)
