from tqdm import tqdm
from konlpy.tag import Komoran as Okt
o = Okt()
inputf = open('C:\\Users\\kjm\\Desktop\\KCCq28_Korean_sentences_UTF8_v2.txt',encoding='utf-8')
count = 0
output = open('okt.txt','w',encoding='utf-8')
for i in inputf:
    count += 1
inputf.seek(0)
for _ in tqdm(range(count)):
    l = inputf.readline()
    l = o.pos(l)
    l = [ '/'.join(ll) for ll in l ]
    l = ' '.join(l)
    output.write(l+'\n')