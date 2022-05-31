import re

def tag_merge(tag_list,token):    
    result = []
    length_count = 0
    pos_result = []
    #print(tag_list)
    pos = []
    for l,tk in zip(tag_list,token):
        l = l.replace('\n','')
        #print('l',l)
        #print('tk',tk)
        if l.startswith('[CLS] '):
            l = l[6:]
        l = l.replace(' O ',' ')
        if l.startswith('O '):
            l = list(l)
            l = l[2:]
            l = ''.join(l)
        l = l.strip()
        l = re.sub(' +',' ',l)
        temp_l = l
        l = l.split(' ')
        tk = tk.replace('\n','')
        tk = tk.strip()
        for ll in l:
            if ll.startswith('B_'):
                result.append(ll[2:])
            else:
                # print(temp_l)
                if len(result) == 0:
                        # result.append(ll.replace('I_',''))
                    result.append('FAIL')
                    # elif result[-1] != ll.replace('I_',''):
                    #     result[-1] = 'FAIL'
        tk = tk.replace('\n','')
        temp_tk = tk
        tk = tk.split(' ')
        result_tk = []
            # print(len(tk),len(result))
        for index,tk_ in enumerate(tk):
            if '+' in tk_:
                result_tk.append(index)
                # print(result)
                try:
                    if (index+1) <= len(result):
                        result[index] = result[index] + '+' + result[index+1]
                        del result[index+1]
                except:
                    l = l
        line_pos_tk = []
        result = ' '.join(result)
        result = result.split(' ')
        #pos.append(result)
        #print(result)
        #print(tk)
        #print(result,tk)
    #for p, tk in zip(pos,token):
        #p = p.split(' ')
        #tk = tk.split(' ')
        for tag,tk_ in zip(result,tk):
            tag__ = tag.split('+')
            tk__ = tk_.split('+')
            temp_pos_tk = []
            for tagg,tkk in zip(tag__,tk__):
                temp_pos_tk.append(tkk+'/'+tagg)
            line_pos_tk.append('+'.join(temp_pos_tk))
        pos_result.append(' '.join(line_pos_tk))
        #print(line_pos_tk)
        result = []
    #    print(p)
    #    print(tk)
    return pos_result
        
