from scipy.io import loadmat
import numpy as np
import cPickle as cp

im = loadmat('matlab.mat')

pics = np.int_(im['mat'])
shu = np.empty([600,102,102])

for i in range(600):
    ar = pics[:,i]
    a = ar.reshape((102,102))
    shu[i,:,:] = a
label = []
for i in range(300):
    label.append('touching')
for i in range(300):
    label.append('not_touching')

d = dict()

d['dataset'] = shu
d['names'] = label

output = open('/home/observer0724/Desktop/Data.pkl', 'wb')

cp.dump(d,output)

output.close()
