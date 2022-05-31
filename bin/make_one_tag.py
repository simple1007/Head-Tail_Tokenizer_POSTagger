def make_one(y):
    temp = []
    for yy in y:
        if '+' not in yy:
            te = yy.split('/')
            yy = te[1].split('_')
            yy = [yyy[0] for yyy in yy]
            temp.append(te[0]+'/'+'_'.join(yy))
        else:
            tt = yy.split('+')
            te = tt[0].split('/')

            y1 = te[1].split('_')
            y1 = [yy1[0] for yy1 in y1]
            y1 = te[0]+'/'+'_'.join(y1)
            
            te = tt[1].split('/')
            y2 = te[1].split('_')
            y2 = [yy2[0] for yy2 in y2]
            y2 = te[0]+'/'+'_'.join(y2)
            temp.append(y1+'+'+y2)
    y = temp

    #print(y)
    return y
