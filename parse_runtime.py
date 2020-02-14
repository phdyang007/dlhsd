import sys
import numpy as np
log = sys.argv[1]
srafmax = int(sys.argv[2])
runtime = []
sraf=[]
iter=[]
with open(log,'r') as f:
    log_all=f.readlines()

for item in log_all:
    if item.startswith("Attack runtime"):
        
        runtime.append(float(item.split(' ')[-1][:-1]))

for item in log_all:
    if item.startswith("ATTACK SUCCEED"):
        
        sraf.append(int(item.split(' ')[4][:-1]))
        iter.append(int(item.split(' ')[-1][:-1]))
sraf=np.asarray(sraf)

print(np.average(np.array(runtime)[np.where(np.array(sraf)<=srafmax)[0]]))

print("max iter is")
print(np.max(np.array(iter)[np.where(np.array(sraf)<=srafmax)[0]]))

print(np.average(runtime))
print(np.average(sraf))