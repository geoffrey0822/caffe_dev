import caffe
import unittest
import tempfile
import numpy as np
import sys,os
import cv2
from caffe import layers as L

def load_net(net_proto):
    f=tempfile.NamedTemporaryFile(mode='w+',delete=False)
    f.write(str(net_proto))
    f.close()
    return caffe.Net(f.name,caffe.TEST)

def simple_network(db_file,mode=''):
    n=caffe.NetSpec()
    pstr='{"scale-to-width":227,"scale-to-height":227,"batch_size":9,"channel":3,"db_file":"%s","mode":"%s"}'%(db_file,mode)
    n.data,n.label=L.Python(name='input',python_param=dict(
                                module='caffe.multilabelDataLayer',
                                layer='MultilabelDataLyer' ,
                                param_str=pstr
                                ),ntop=2)
    return n.to_proto()

mode=''
if len(sys.argv)>2:
    mode=sys.argv[2]
net_proto=simple_network(sys.argv[1],mode)
net=load_net(net_proto)
net.forward()
print net.blobs['data'].shape
print net.blobs['label'].shape
print np.array(net.blobs['data'].shape)
print np.array(net.blobs['label'].shape)
print net.blobs['label']
nBlob=np.array(net.blobs['data'].shape)[0]
for n in range(nBlob):
    print 'mother class: %d'%net.blobs['label'].data[n,...]
img=np.zeros(np.array(net.blobs['data'].shape)[1:],dtype=np.float32)
img.data=net.blobs['data'].data[0,...]
img=img.swapaxes(0,2).swapaxes(1,0)
cv2.imwrite('/home/gathetaroot/testLayer.png',img)
