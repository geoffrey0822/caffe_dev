import caffe
import os,sys
import numpy as np
import cv2
import json
import csv

class MultilabelDataLyer(caffe.Layer):
    def setup(self,bottom,top):
        params=json.loads(self.param_str)
        self.db_file=''
        if 'db_file' in params:
            self.db_file=params['db_file']
        else:
            print 'the database file must be given'
            raise
        self.width=0
        self.height=0
        if 'scale-to-width' in params:
            self.width=int(params['scale-to-width'])
        if 'scale-to-height' in params:
            self.height=int(params['scale-to-height'])
        self.nlabel=1
        self.nchannel=1
        self.batch=1
        if self.height>0 and self.width>0 and 'batch_size' in params:
            self.batch=int(params['batch_size'])
        if len(top)!=2:
            print 'MultilabelDataLayer contains 2 outputs'
            raise
        if len(bottom)>0 and bottom!=[]:
            print 'No input supported for the Data Layer'
            raise 
        self.total_rec=0
        with open(self.db_file,'r') as db:
            reader=csv.reader(db,delimiter=',')
            for row in reader:
                self.total_rec+=1
                if self.total_rec==1:
                    print row
                    self.nlabel=len(row[1].split(';'))
                    self.nchannel=cv2.imread(row[0]).shape[2]
        self.current_batch=0
        self.max_batch=int(np.floor(self.total_rec/self.batch))
        if self.batch>1:
            top[0].reshape(self.batch,self.nchannel,self.height,self.width)
            top[1].reshape(self.batch,self.nlabel)
        self.db=open(self.db_file,'r')
        self.reader=csv.reader(self.db,delimiter=',')
    
    def reshape(self,bottom,top):
        if self.batch>1:
            if self.current_batch>=self.max_batch:
                remain=self.total_rec-self.batch*self.current_batch
                top[0].reshape(remain,self.nchannel,self.height,self.width)
                top[1].reshape(remain,self.nlabel)
                self.current_batch=0
            elif self.current_batch==0:
                top[0].reshape(self.batch,self.nchannel,self.height,self.width)
                top[1].reshape(self.batch,self.nlabel)
        else:
            record=self.next()
            shape=cv2.imread(record[0]).shape
            top[0].reshape(1,self.nchannel,shape[0],shape[1])
            top[1].reshape(1,self.nlabel)
    
    def forward(self,bottom,top):
        if self.batch>1:
            startAt=self.current_batch*self.batch
            limit=self.batch
            if self.current_batch>=self.max_batch:
                limit=self.total_rec-self.batch*self.current_batch
                csv.reader(self.db,delimiter=',')
            for nbatch in range(limit):
                #record=self.reader[startAt+nbatch]
                record=self.reader.next()
                img=cv2.imread(record[0])
                if self.batch>1:
                    img=cv2.resize(img,(self.width,self.height))
                img=img[:,:,(2,1,0)]
                img=img.swapaxes(0,2).swapaxes(1,2)
                top[0].data[nbatch,...]=img
                top[1].data[nbatch,...]=record[1].split(';')
        self.current_batch+=1
    
    def backward(self,top,propagate_down,bottom):
        pass