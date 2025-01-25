from collections import defaultdict

W_max = 200
W_min = 2


user_dict = defaultdict(dict)
def get_symbol(c):
    x = c%3
    if x == 0:
        return 'B'
    elif x == 1:
        return 'A'
    else :
        return 'T'
with open('./user_head.txt', 'r') as fin:
#with open('./test.txt', 'r') as fin:
    cnt = 0
    for line in fin:
        line = line.strip().split("\t")
        # B/A/T
        symbol = get_symbol(cnt)
        user_id = line[0]
        slist = line[1:]
        user_dict[user_id][symbol] = slist
        cnt += 1
print('Len of UserDict', len(user_dict))


def write_ins(f, tp):
    uid, bss, ass, rank, label = tp
    # 前期不够W_max，b补0，a补jjj
    if len(bss) < W_max:
        bss = ['0']* (W_max - len(bss)) + bss
    if len(ass) < W_max:
        ass = ['1']* (W_max - len(ass)) + ass

    b,a,t = label
    f.write('\t'.join([  uid, ','.join(bss), ','.join(ass), str(rank), a, b, t]) + '\n')

with open('./user.ins', 'w') as fout:
  for uid, ldict in user_dict.items():
    B = ldict.get('B')
    A = ldict.get('A')
    T = ldict.get('T')
    cur_bs = []
    cur_as = []
  
    # each user
    rank = 0
    for b, a, t in zip(B,A,T):
        b = str(int(min(int(b), 100000)/100)) # 单位是cents
        a = str(min(float(a), 2000.0)) # 单位是k
        rank += 1
        if len(cur_bs) >= W_min:
            write_ins(fout, (uid, cur_bs, cur_as, rank, (b,a,t)))
        if len(cur_bs) == W_max:
          cur_bs.pop()
          cur_as.pop()
        cur_bs.append(b)
        cur_as.append(a)
    if len(cur_bs) > W_min:
       rank += 1
       # 把最后窗口内的ins拿完
       if t  == '0':
           write_ins(fout, (uid, cur_bs, cur_as,rank, (b,a,t)))
       else:
           write_ins(fout, (uid, cur_bs, cur_as,rank, ('0', '1', '1')))