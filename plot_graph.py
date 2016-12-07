from matplotlib import pyplot as plt


op = {}
op[0] = 'tf.train.GradientDescentOptimizer'
op[1] = 'tf.train.AdadeltaOptimizer'
op[2] = 'tf.train.AdagradOptimizer'
# op[3] = 'tf.train.AdagradDAOptimizer'
for k in range(3):
    l = {}
    plt.close('all')
    for j in range(5):
        l[j] = []
        patch = []
        for i in range(10):
            depth1 = 1+j
            depth2 = 3+j
            patch_size = 2+i
            patch.append(patch_size)
            folder = str(patch_size)+'_'+str(patch_size)+'with'+str(depth1)+'and'+str(depth2)+'_'+op[k]
            f=open('C:/Users/mizuki nana/Desktop/31_31_images/'+folder+'/'+folder+'_record.txt','r')
            lines = f.readlines()
            l[j].append(float(lines[300][15:].strip('%\n')))
        print (len(l[j]))
        print (l[j])
        plt.plot(patch,l[j], label = str(depth1)+str(depth2))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(op[k]+str(depth1)+'and'+str(depth2)+'.png')
# plt.show()
