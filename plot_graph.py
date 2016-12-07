from matplotlib import pyplot as plt
op = {}
op[0] = 'tf.train.GradientDescentOptimizer'
op[1] = 'tf.train.AdadeltaOptimizer'
op[2] = 'tf.train.AdagradOptimizer'
op[3] = 'tf.train.AdagradDAOptimizer'
l = []
for i in range(10):
    depth1 = 1
    depth2 = 3
    patch_size = 2+i
    folder = str(patch_size)+'*'+str(patch_size)+'with'+str(depth1)+'and'+str(depth2)+'_'+op[0]
    f=open('/home/yida/Desktop/da_vincii/31*31_images/'+folder+'/'+folder+'_record.txt','rb')
    lines = f.readlines()
    l.append(float(lines[300][15:].strip('%\n')))
plt.plot(patch,l)
plt.savefig(op[0]+str(depth1)+'and'+str(depth2)+'.jpg')
