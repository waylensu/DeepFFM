#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)
import os.path as osp

pre_path = '/Users/wing/DataSet/criteo'
train_path = osp.join(pre_path, 'train.txt')
test_path = osp.join(pre_path, 'test.txt')

save_path = '/Volumes/Untitled/data_set/pre'
train_ind_path = osp.join(save_path, 'train_ind.txt')
train_val_path = osp.join(save_path, 'train_val.txt')
train_label_path = osp.join(save_path, 'train_label.txt')
#test_ind_path = osp.join(save_path, 'test_ind.txt')
#test_val_path = osp.join(save_path, 'test_val.txt')
#test_label_path = osp.join(save_path, 'test_label.txt')
limits_path = osp.join(save_path, 'limits.txt')

int_max = [0] * 13
int_min = [0] * 13
cate_table = [{} for i in range(26)]

def get_cate_ind(i, cate, is_train):
    if cate in cate_table[i]:
        return cate_table[i][cate], 1
    elif is_train:
        cate_table[i][cate] = len(cate_table[i])
        return cate_table[i][cate], 1
    else:
        return 0,0

def transform(src_path, ind_path, val_path, label_path, is_train = True, ind_len = 13, cate_len = 26):
    with open(src_path) as in_file:
        with open(ind_path, 'w') as ind_file:
            with open(val_path, 'w') as val_file:
                with open(label_path, 'w') as label_file:
                    for line in in_file:
                        cols = line.strip('\n').split('\t')
                        label_file.write(cols[0]+'\n')
                        inds = [0] * (ind_len + cate_len)
                        vals = [0] * (ind_len + cate_len)
                        for i in range(ind_len):
                            col = cols[i+1]
                            if not col == '':
                                col = int(col)
                                vals[i] = col
                                if col > int_max[i]:
                                    int_max[i] = col
                                elif col < int_min[i]:
                                    int_min[i] = col
                        for i in range(cate_len):
                            col = cols[i+ind_len+1]
                            if col == '':
                                vals[i+ind_len] = 0
                            else:
                                inds[i+ind_len], vals[i+ind_len] = get_cate_ind(i, col, is_train)
                        ind_file.write('\t'.join(map(str,inds)) + '\n')
                        val_file.write('\t'.join(map(str,vals)) + '\n')


transform(train_path, train_ind_path, train_val_path, train_label_path)
#transform(test_path, test_ind_path, test_val_path, test_label_path, False)

with open(limits_path, 'w') as out_file:
    out_file.write('\t'.join(map(str,int_max)) + '\n')
    out_file.write('\t'.join(map(str,int_min)) + '\n')
    out_file.write('\t'.join([str(len(x)) for x in cate_table]) + '\n')
