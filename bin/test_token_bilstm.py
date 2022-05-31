import tensorflow as tf
import numpy as np
import pickle
import argparse
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,default='test_data_x.txt')
parser.add_argument('--output',type=str,default='tokenizer.txt')
parser.add_argument('--mode',type=str,default='bilstm')
parser.add_argument('--maxline',type=int,default=50000)

args = parser.parse_args()
if args.mode == 'bilstm':
    model = tf.keras.models.load_model('lstm_bigram_tokenizer.model',compile=False)
elif args.mode == 'bert':
    model = tf.keras.models.load_model('bert_headtail_tokenizer.model',compile=False)

if args.mode == 'bilstm':
    with open('lstm_vocab.pkl','rb') as f:
        vocab = pickle.load(f)
    with open('bigram_vocab.pkl','rb') as f:
        bigram = pickle.load(f)
elif args.mode == 'bert':
    with open('eum_vocab.pkl','rb') as f:
        vocab = pickle.load(f)
max_len = 300
import time
start = time.time()
# print(vocab.keys())
# x = '안녕하세요 국민대학▁김정민이다▁.'
# x = '내▁눈을▁본다면▁밤하늘의▁별이▁되는▁기분을▁느낄▁수▁있을▁거야'
# x = '나는▁밥을▁먹고▁집에▁갔다.'

#def make_input(x_t):
#	masks = []
#	segments = []

#	for x in x_t:
		# for d in x:
#		d_ = [t for t in x if t != 1]
		# print(len(d_))
#		mask = [1]*len(d_) + [0] * (max_len-len(d_))
#		masks.append(mask)
#		segment = [0] * max_len
#		segments.append(segment)
		# masks = masks * 100
		# segments = segments * 100
#	masks = np.array(masks)
#	segments = np.array(segments)

#	return masks,segments

def w2i(x_t):
	res = []
	for i in x_t:
		x = i.replace('  ',' ').replace(' ','▁')
		x = list(x)
		x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 

		x = x

		x = x + [0] * (max_len-len(x))
		res.append(x)

	x = np.array(res)

	return x
def make_input(x_t):
	masks = []
	segments = []

	for x in x_t:
		d_ = [t for t in x if t != 1]
		mask = [1]*len(d_) + [0] * ((max_len+2)-len(d_))
		masks.append(mask)
		segment = [0] * (max_len+2)
		segments.append(segment)
	masks = np.array(masks)
	segments = np.array(segments)

	return masks,segments

def w2i_bert(x_t):       
	res = []
	for i in x_t:
		x = i.replace('  ',' ').replace(' ','▁')
		x = list(x)
		x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ]
		x = [2]+x+[3]
		x = x + [1] * ((max_len+2)-len(x))
		res.append(x)

	x = np.array(res)
	return x

def predict(x):
	# x = x.replace('  ',' ').replace(' ','▁')
	# x = list(x)
	BI = []
	x_temp = x
	x_len = len(x)
	if args.mode == 'bert':
		x = w2i_bert(x)
	elif args.mode == 'bilstm':
		for l in x:
			bi_npy = []
			for i in range(len(l)-1):
				bi = l[i:i+2]
				if bi in bigram:
					bi_npy.append(bigram[bi])
				else:
					bi_npy.append(bigram['[UNK]'])
			bi_npy = bi_npy + [0] * (max_len - len(bi_npy))
			bi_npy = np.array(bi_npy)
			BI.append(bi_npy)
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
	if args.mode == 'bilstm':
		BI = np.array(BI)
		result = model.predict([x,BI])
	elif args.mode  == 'bert':
		result = model.predict([x,masks,segments])
	result_ = tf.argmax(result,axis=2).numpy()
	tagging = []
	if args.mode == 'bert':
		result_ = result_[:,1:]
	for index,tag_ in enumerate(result_):
		x_te = list(x_temp[index])
		for index_temp,te in enumerate(tag_):
			if te == 2:
				x_te[index_temp] = x_te[index_temp] + '+'
			elif te == 3 and index_temp < len(tag_)-1 and tag_[index_temp+1] == 3:
				break
		x_te = ''.join(x_te).replace('▁',' ')
		x_te = x_te.split(' ')
		temp_tok = []
		for xttt in x_te:
			if xttt.count('+') > 1:
				xttt = xttt.split('+')
				h = xttt[0]
				t = ''.join(xttt[1:])
				xttt = h+'+'+t
			temp_tok.append(xttt)
		x_te = ' '.join(temp_tok)
		tagging.append(x_te)
	# print(result.shape)
	# for index,i in enumerate(result):
	# 	array = []
	# 	stop = -1
	# 	# for re in range(len(i)-1,0,-1):
	# 	# 	# print(result[re])
	# 	# 	tag = tf.argmax(i[re])
	# 	# 	# array.append(tag)

	# 	# 	if tag != 3:
	# 	# 		stop = re
	# 	# 		break
	# 	# temp = i[:stop+1]
	# 	temp = i
	# 	x_te = list(x_temp[index])
	# 	for index_temp,te in enumerate(temp):
	# 		tag = tf.argmax(te)
	# 		if tag.numpy() == 2:
	# 			#array.append(index_temp)
	# 			x_te[index_temp] = x_te[index_temp] + '<tab>'
	# 			# print(index,tag.numpy())
	# 			# print(x_te)
	# 		elif tag.numpy() == 3 and index_temp < len(temp)-1 and tf.argmax(temp[index_temp+1]) == 3:
		# 		break
		# 	elif tag.numpy() == 1:
		# 	   x_temp[index_temp] = x_temp[index_temp] + '▁'
		# print(tagging)
		# tagging.append(''.join(x_te).replace('▁',' '))
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

#result_file_y = open(args.mode+'_'+args.output.replace('.txt','')+'_y.txt','w',encoding='utf-8')
result_file = open(args.mode+'_'+args.output,'w',encoding='utf-8')

#result_file = open(args.mode+'_'+args.output,'w',encoding='utf-8')
from tqdm import tqdm
import re
import subprocess
with open(args.input,'r',encoding='utf-8') as f:
	#with open('y_train.txt','r',encoding='utf-8') as ff:
		count = 0
		X = []
		Y = []
		#for l in f:
		dirr =os.getcwd()
		#print(dirr)
		maxline = int(subprocess.check_output("wc -l "+dirr+'/'+args.input,shell=True).split()[0])
		args.maxline = maxline
		# print(dirr+'/'+args.input)
		for _ in tqdm(range(args.maxline)):
			l = f.readline()
			l = l.replace('\n','')
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
			#yy = ff.readline()
			# print(l)
			#l = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', ' ', l)
			#l = re.sub('[ㄱ-ㅎ]',' ',l)
			#l = re.sub('[ㅏ-ㅣ]',' ',l)
			l = re.sub(' +',' ',l)
			#print(l)
			#tt = l.strip().replace(' ','')
			#y = l.split('\t')[1]
			#l = l.split('\t')[0]
			#print(y)
			#l = ' '.join(l)
			#y = l.split(' ')[-1]
			if (args.mode == 'bilstm' and len(l) <= 300) or (args.mode=='bert' and len(l) <= 300): #and len(tt) > 1:
                            
				l = re.sub(' +',' ',l)
				X.append(l.replace('  ',' ').replace(' ','▁'))
				#Y.append(y)
				count += 1
				# print(count)
				if count % 100 ==0:
					result = predict(X)#.replace('▁',' ')
					result_file.write('\n'.join(result)+'\n')
					#result_file_y.write('\n'.join(Y)+'\n')
					Y = []
					X = []
				# import sys
				# sys.exit()
			else:

				l=l#result_file_y.write(l+'\n')

if len(X) > 0:
	result = predict(X)#.replace('▁ ',' ')
	result_file.write('\n'.join(result)+'\n')   
	#result_file_y.write('\n'.join(Y)+'\n')
result_file.close()
#result_file_y.close()
end = time.time()

print(end-start)
