#!/bin/bash


# =============================================================
#  File Name : a.sh
#  Author : waylensu
#  Mail : waylensu@163.com
#  Created Time : 2017年04月02日 星期日 12时48分07秒
# =============================================================

date=`date "+%Y-%m-%d-%H-%M-%S"`
nohup python3 startFFM.py >log/$date.log 2>&1  &
