import os,sys,leveldb
import numpy as np
import caffe

class WordDataLayer(caffe.Layer):
    def setup(self,bottom,top):
        if not bottom is None and len(bottom)>0:
            raise exception('Data Layer not support for input')
        if not len(top)==2:
            raise exception('The number of output must be 2')
        params=eval(self.param_str)
        db_path=params['dataset']
        self.batch=1
        if 'batch_size' in params:
            self.batch=int(params['batch_size'])
        self.db=leveldb.LevelDB(db_path)
        self.current_idx=0
        self.ndata=0
        top[0].reshape(self.batch,1)
        top[1].reshape(self.batch,1)
	self.data_size=0
	if 'data_size' in params:
	    self.data_size=int(params['data_size'])
	else:
            for key,value in self.db.RangeIter():
                self.ndata+=1
    
    def reshape(self,bottom,top):
        pass
    
    def forward(self,bottom,top):
        for i in range(self.batch):
            data=np.frombuffer(self.db.Get(str(self.current_idx),default=None))
            top[0].data[i,...]=data[0]
            top[1].data[i,...]=data[1]
            self.current_idx+=1
            if self.current_idx>=self.ndata:
                self.current_idx=0
    
    def backward(self,top,propagate_doen,bottom):
        pass
