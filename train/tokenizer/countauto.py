with open('ht_test_x.txt',encoding='utf-8') as f:
    with open('ht_test_y.txt',encoding='utf-8') as ff:
        count = 0
        for l, ll in zip(f,ff):
            if len(l) != len(ll):
                count += 1

print(count)
