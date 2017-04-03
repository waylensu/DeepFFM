#coding=utf8
from __future__ import (division,absolute_import,print_function,unicode_literals)
import os.path as osp


save_path = '/home/wing/DataSet/criteo/pre'
train_ind_path = osp.join(save_path, 'train_ind.txt')
train_val_path = osp.join(save_path, 'train_val.txt')
train_label_path = osp.join(save_path, 'train_label.txt')
train_ffm_path = osp.join(save_path, 'train_ffm.txt')
limits_path = osp.join(save_path, 'limits.txt')

with open(limits_path) as in_file:
    cols = in_file.readline().strip().split('\t')
    lens = list(map(int,cols))
    offsets = [0]
    for l in lens:
        offsets.append(offsets[-1] + l)



with open(train_ffm_path, 'w') as ffm_file:
    with open(train_ind_path) as ind_file:
        with open(train_val_path) as val_file:
            with open(train_label_path) as label_file:
                for ind_line, val_line, label in zip(ind_file, val_file, label_file):
                    line = label.strip()
                    inds = ind_line.strip().split('\t')
                    vals = val_line.strip().split('\t')
                    for field,(ind, val, offset) in enumerate(zip(inds, vals, offsets)):
                        line += ' ' + str(field+1) + ':' + str(1 + offset + int(ind)) + ':' + val
                    ffm_file.write(line+'\n')

