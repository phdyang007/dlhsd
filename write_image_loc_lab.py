

import os
import sys

vallayout  = []
trainlayout = []
testlayout = []
'''
WRITING TEST INPUT TXT
'''
print ("Usage (e.g.):")
print ("python write_image_loc_lab.py ~/benchmark_iccad/ICCAD1/train ~/caffe-hsd/models/ICCAD1/train.txt N")
print ("python write_image_loc_lab.py ~/benchmark_iccad/ICCAD1/val ~/caffe-hsd/models/ICCAD1/val.txt N")
print ("N>=1 is upsampling number.")
argv = sys.argv

for dirname, dirnames, filenames in os.walk(argv[1]):
    for i in range(0, len(filenames)):
        testlayout.append([])
        testlayout[i].append(os.path.join(dirname,filenames[i]))
        if filenames[i][0] == 'N':
            testlayout[i].append(0)
        else:
            testlayout[i].append(1)

    # print path to all subdirectories first.
 #   for subdirname in dirnames:
 #       print(os.path.join(dirname, subdirname))

    # print path to all filenames.
 #   for filename in filenames:
 #       print(os.path.join(dirname, filename))
N=int(argv[3])
with open(argv[2], 'w') as t:
    for i in range(0, len(testlayout)):
        if testlayout[i][1]==1:
            for j in range(0,N):
                t.write(testlayout[i][0])
                t.write(' ')
                t.write(str(testlayout[i][1]))
                t.write('\n')
        else:
            t.write(testlayout[i][0])
            t.write(' ')
            t.write(str(testlayout[i][1]))
            t.write('\n')

    
