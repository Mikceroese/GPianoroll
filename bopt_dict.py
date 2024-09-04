#################################################
# This file contains the BayesOptDict class     #
#                                               #
# The objective of this class is to provide a   #
# way to manipulate the parameters and samples  #
# of a BayesOpt object with Python.             #
#                                               #
# The main functionalities include creating a   #
# BayesOpt-like object from a saved text file   #
# containing the model and vice versa.          #
# (Note: You can't create the object from       #
# scratch, you must read a textfile.)           #
#                                               #
# Author: Miguel Marcos (m.marcos@unizar.es)    #
#################################################

# Import zone

import numpy as np

class BayesOptDict():

    def __init__(self, txt_path):

        # All parameters present in the file are saved as a list
        # A dictionary with corresponding parameter name <-> index
        # association is also saved.
        self.params = []
        self.param_idxs = {}


        # Data points, scores and debug points are also saved separately.
        self.mY = []
        self.yShape = []
        self.mX = []
        self.xShape = []
        self.mDebug = []
        self.debugShape = []

        with open(txt_path) as f:
            lines = f.readlines()

        for i,line in enumerate(lines):

            chunks = line.split('=')

            # If the list contains more than one chunk, it's a parameter
            if len(chunks)>1:
                self.params.append((chunks[0],chunks[1]))
                self.param_idxs[chunks[0]]=i
                if chunks[0] == "mY":
                    chunks = chunks[1].split(']')
                    self.yShape = np.array(int(chunks[0][1:]))
                    self.mY = np.fromstring(chunks[1][1:-2],dtype=float,sep=',').reshape(self.yShape)
                elif chunks[0] == "mX":
                    chunks = chunks[1].split(']')
                    self.xShape = np.fromstring(chunks[0][1:],dtype=int,sep=',')
                    self.mX = np.fromstring(chunks[1][1:-2],dtype=float,sep=',').reshape(self.xShape)
                elif chunks[0] == "mDebugBestPoints":
                    chunks = chunks[1].split(']')
                    self.debugShape = np.fromstring(chunks[0][1:],dtype=int,sep=',')
                    self.mDebug = np.fromstring(chunks[1][1:-2],dtype=float,sep=',').reshape(self.debugShape)

    def save_txt(self, txt_path):

        with open(txt_path,'w') as f:

            for param in self.params:

                if param[0] == "mY":
                    mY = (str(tuple(self.mY.flatten())))
                    if mY[-2] == ',':
                        mY = mY[:-2]+")"
                    line = param[0]+"=["+str(self.yShape)+"]"+mY.replace(' ','')+'\n'
                    f.write(line)
                elif param[0] == "mX":
                    mX = (str(tuple(self.mX.flatten())))
                    if mX[-2] == ',':
                        mX = mX[:-2]+")"
                    line = param[0]+"="+np.array2string(self.xShape,precision=4,separator=',')+mX.replace(' ','')+'\n'
                    f.write(line)
                elif param[0] == "mDebugBestPoints":
                    mDebug = (str(tuple(self.mDebug.flatten())))
                    if mDebug[-2] == ',':
                        mDebug = mDebug[:-2]+")"
                    line = param[0]+"="+np.array2string(self.debugShape,precision=4,separator=',')+mDebug.replace(' ','')+'\n'
                    f.write(line)
                else:
                    line = param[0]+'='+param[1]
                    f.write(line)

    def order_samples(self):

        order = np.argsort(self.mY)
        self.mY = self.mY[order]
        self.mX = self.mX[order]

    def get_sample(self,idx):

        return (self.mX[idx],self.mY[idx])

    def add_sample(self,Xin,Y):

        self.xShape[0] += 1
        self.mX = np.vstack((self.mX,Xin))
        self.yShape += 1
        self.mY = np.append(self.mY,Y)

    def set_score(self,idx,score):

        mY[idx] = score

    def set_init_samples(self,n_init_samples):

        self.params[self.param_idxs['mParameters.n_init_samples']] = ('mParameters.n_init_samples',str(n_init_samples)+'\n')

    def set_num_iter(self,n_iter):

        self.params[self.param_idxs['mParameters.n_iterations']] = ('mParameters.n_iterations',str(n_iter)+'\n')
