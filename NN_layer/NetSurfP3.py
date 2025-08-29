# --*-- conding:utf-8 --*--
# @time:8/28/25 23:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:NetSurfP3.py

import biolib, pathlib

nsp3 = biolib.load('DTU/NetSurfP-3')

job = nsp3.run(input_file='seqs/my_protein.fasta')


print("Outputs at:", job.output_path)
print("Files:", [p.name for p in pathlib.Path(job.output_path).iterdir()])
