import caffe
import unittest
import tempfile
import numpy as np
import sys,os
from caffe import layers as L

def load_net(net_proto):
    f=tempfile.NamedTemporaryFile(mode='w+',delete=False)
    f.write(str(net_proto))
    f.close()
    return caffe.Net(f.name,caffe.TEST)

def simple_network(db_file):
    n=caffe.NetSpec()
    pstr='{"scale-to-width":227,"scale-to-height":227,"batch_size":32,"channel":3,"db_file":"%s"}'%db_file
    n.data,n.label=L.Python(name='input',python_param=dict(
                                module='caffe.multilabelDataLayer',
                                layer='MultilabelDataLyer' ,
                                param_str=pstr
                                ),ntop=2)
    return n.to_proto()

net_proto=simple_network(sys.argv[1])
net=load_net(net_proto)
net.forward()
print net.blobs['data'].shape
print net.blobs['label'].shape
print net.blobs['label']
