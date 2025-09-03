# --*-- conding:utf-8 --*--
# @time:9/3/25 15:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test.py

import biolib

if __name__ == '__main__':

    netsurfp_3 = biolib.load('DTU/NetSurfP-3')
    print(netsurfp_3.cli(args='--help'))