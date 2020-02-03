import numpy as np
from utils import sigmoid,softmax
from dataloader import idx_to_char,char_to_idx,dataset

class LSTM(object):

    def __init__(self,lr=1e-1,time_steps=25,len_of_vocab=25,mean=0.,std=0.01,mode='lstm'):
        self.input = None
        self.output = None
        self.lr = lr
        self.time_steps = time_steps
        self.len_of_vocab = len_of_vocab
        self.mean = mean
        self.std = std
        self.mode = mode
        self.loss = 0
        self.backward_flag = False
        self.Wi,self.Wf,self.Wz,self.Wo,self.Wout = np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                            np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                            np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                            np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                            np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab))

        self.Ri,self.Rf,self.Rz,self.Ro = np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                        np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                        np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab)),\
                        np.random.normal(self.mean,self.std,(self.len_of_vocab,self.len_of_vocab))

        self.Pi,self.Pf,self.Po = np.random.normal(self.mean,self.std,(self.len_of_vocab,1)),\
                    np.random.normal(self.mean,self.std,(self.len_of_vocab,1)),\
                    np.random.normal(self.mean,self.std,(self.len_of_vocab,1))

        self.bi,self.bo,self.bf,self.bz,self.bout = np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1))

        self.mWi,self.mWf,self.mWz,self.mWo,self.mWout = np.zeros_like(self.Wi),\
            np.zeros_like(self.Wf),\
            np.zeros_like(self.Wz),\
            np.zeros_like(self.Wo),\
            np.zeros_like(self.Wout)

        self.mRi,self.mRf,self.mRz,self.mRo = np.zeros_like(self.Ri),np.zeros_like(self.Rf),np.zeros_like(self.Rz),np.zeros_like(self.Ro)

        self.mPi,self.mPf,self.mPo = np.zeros_like(self.Pi),np.zeros_like(self.Pf),np.zeros_like(self.Po)

        self.mbi,self.mbo,self.mbf,self.mbz,self.mbout = np.zeros_like(self.bi),np.zeros_like(self.bo),np.zeros_like(self.bf),np.zeros_like(self.bz),np.zeros_like(self.bout)
        
        self.dWi,self.dWf,self.dWz,self.dWo,self.dWout = np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab))

        self.dRi,self.dRf,self.dRz,self.dRo = np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab))

        self.dPi,self.dPf,self.dPo  = np.zeros((self.len_of_vocab,1)),\
                    np.zeros((self.len_of_vocab,1)),\
                    np.zeros((self.len_of_vocab,1))

        self.dbi,self.dbo,self.dbf,self.dbz,self.dbout = np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1))
        
        self.hs={}
        self.cs={}
        self.i_gate={}
        self.f_gate={}
        self.o_gate={}
        self.z = {}
        self.p = {}
                            
    
    def zero_grad(self):
        self.dWi,self.dWf,self.dWz,self.dWo,self.dWout = np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                            np.zeros((self.len_of_vocab,self.len_of_vocab))

        self.dRi,self.dRf,self.dRz,self.dRo = np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab)),\
                        np.zeros((self.len_of_vocab,self.len_of_vocab))

        self.dPi,self.dPf,self.dPo  = np.zeros((self.len_of_vocab,1)),\
                    np.zeros((self.len_of_vocab,1)),\
                    np.zeros((self.len_of_vocab,1))

        self.dbi,self.dbo,self.dbf,self.dbz,self.dbout = np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1)),\
                            np.zeros((self.len_of_vocab,1))

        self.backward_flag = False

    def clip_grad(self,clip_val=1):
            for dparam in [self.dWi,self.dWf,self.dWz,self.dWo,self.dWout,self.dRi,self.dRf,self.dRz,self.dRo,self.dPi,self.dPo,self.dPf,self.dbi,self.dbo,self.dbf,self.dbz,self.dbout]:
                np.clip(dparam,-clip_val,clip_val,out=dparam)
        
    def step(self):
        for params,dparam,mparam in zip([self.Wi,self.Wf,self.Wz,self.Wo,self.Wout,self.Ri,self.Rf,self.Rz,self.Ro,self.Pi,self.Po,self.Pf,self.bi,self.bo,self.bf,self.bz,self.bout],\
		[self.dWi,self.dWf,self.dWz,self.dWo,self.dWout,self.dRi,self.dRf,self.dRz,self.dRo,self.dPi,self.dPo,self.dPf,self.dbi,self.dbo,self.dbf,self.dbz,self.dbout],\
		[self.mWi,self.mWf,self.mWz,self.mWo,self.mWout,self.mRi,self.mRf,self.mRz,self.mRo,self.mPi,self.mPo,self.mPf,self.mbi,self.mbo,self.mbf,self.mbz,self.mbout]):
            mparam += dparam*dparam
            params += -self.lr*dparam/np.sqrt(mparam+1e-8)

    def set_input(self,input,output):
        self.input = input
        self.output = output

    
    def sample(self,h_prev,c_prev,num_char):
        hs = np.copy(h_prev)
        cs = np.copy(c_prev)
        x = np.zeros((self.len_of_vocab,1))
        x[np.random.randint(0,self.len_of_vocab),0] = 1
        idxs = []
        for _ in range(num_char):

            I = np.dot(self.Wi,x) + np.dot(self.Ri,hs) + self.Pi*cs + self.bi
            i_gate = sigmoid(I)

            F = np.dot(self.Wf,x) + np.dot(self.Rf,hs) + self.Pf*cs + self.bo
            f_gate = sigmoid(F)

            Z = np.dot(self.Wz,x) + np.dot(self.Rz,hs) + self.bz
            z = np.tanh(Z)

            cs = i_gate*z + f_gate*cs

            O = np.dot(self.Wo,x) + np.dot(self.Ro,hs) + self.Po*cs +self.bo
            o_gate = sigmoid(O)

            hs = o_gate * np.tanh(cs)

            if self.mode == 'lstm':
                out = np.dot(self.Wout,hs) + self.bout
            
            if self.mode == 'blstm':
                out = np.dot(self.Wout,hs)

            p = softmax(out)
            idx = np.random.choice(self.len_of_vocab,1,p=p.ravel())[0]
            x = np.zeros((self.len_of_vocab,1))
            x[idx,0] = 1
            idxs.append(idx)

        print(''.join(idx_to_char[c] for c in idxs))

    #forward_backward_pass
    def forward(self,h_prev,c_prev):
        self.hs={}
        self.cs={}
        self.i_gate={}
        self.f_gate={}
        self.o_gate={}
        self.z ={}
        self.hs[-1] = np.copy(h_prev)
        self.cs[-1] = np.copy(c_prev)
        self.p = {}
        self.loss = 0
        for t in range(self.time_steps):
            x = np.zeros((self.len_of_vocab,1))
            x[self.input[t],0] = 1

            I = np.dot(self.Wi,x) + np.dot(self.Ri,self.hs[t-1]) + self.Pi*self.cs[t-1] + self.bi
            self.i_gate[t] = sigmoid(I)

            F = np.dot(self.Wf,x) + np.dot(self.Rf,self.hs[t-1]) + self.Pf*self.cs[t-1] + self.bf
            self.f_gate[t] = sigmoid(F)

            Z = np.dot(self.Wz,x) + np.dot(self.Rz,self.hs[t-1]) + self.bz
            self.z[t] = np.tanh(Z)

            self.cs[t] = self.i_gate[t]*self.z[t] + self.f_gate[t]*self.cs[t-1]

            O = np.dot(self.Wo,x) + np.dot(self.Ro,self.hs[t-1]) + self.Po*self.cs[t] +self.bo
            self.o_gate[t] = sigmoid(O)

            self.hs[t] = self.o_gate[t] * np.tanh(self.cs[t])

            if self.mode == 'lstm':
                out = np.dot(self.Wout,self.hs[t]) + self.bout
                self.p[t] = softmax(out)
                self.loss += -np.log(self.p[t][self.output[t],0])
                

    #Backward pass
    def backward(self,bdout=None):
        dht_z = np.zeros((self.len_of_vocab,1))
        dht_f = np.zeros((self.len_of_vocab,1))
        dht_o = np.zeros((self.len_of_vocab,1))
        dht_i = np.zeros((self.len_of_vocab,1))

        dct_cs = np.zeros((self.len_of_vocab,1))
        dct_f = np.zeros((self.len_of_vocab,1))
        dct_o = np.zeros((self.len_of_vocab,1))
        dct_i = np.zeros((self.len_of_vocab,1))


        if self.mode == 'lstm':
            for t in reversed(range(self.time_steps)):
                x = np.zeros((self.len_of_vocab,1))
                x[self.input[t],0] = 1
                
                dout = np.copy(self.p[t])
                dout[self.output[t],0] -= 1
                self.dWout += np.dot(dout,self.hs[t].T)
                dht = np.dot(self.Wout.T,dout) + dht_z + dht_f + dht_o + dht_i
                
                self.dbout += dout
            
                dog = np.tanh(self.cs[t])*dht
                dog_ = self.o_gate[t]*(1-self.o_gate[t])*dog
                self.dWo += np.dot(dog_,x.T)
                self.dRo += np.dot(dog_,self.hs[t-1].T)
                dht_o = np.dot(self.Ro.T,dog_)
                self.dPo += self.cs[t]*dog_
                dct_o = self.Po * dog_
                self.dbo += dog_

                dct = (1-np.tanh(self.cs[t])*np.tanh(self.cs[t]))*self.o_gate[t]*dht + dct_cs + dct_f + dct_o + dct_i
                dig = self.z[t] * dct
                dz  = self.i_gate[t] * dct
                dfg = self.cs[t-1] * dct
                dct_cs = self.f_gate[t] * dct

                dz_ = (1-self.z[t]*self.z[t])*dz
                self.dWz += np.dot(dz_,x.T)
                self.dRz += np.dot(dz_,self.hs[t-1].T)
                dht_z = np.dot(self.Rz.T,dz_)
                self.dbz += dz_

                dfg_ = self.f_gate[t]*(1-self.f_gate[t])*dfg
                self.dWf += np.dot(dfg_,x.T)
                self.dRf += np.dot(dfg_,self.hs[t-1].T)
                dht_f = np.dot(self.Rf.T,dfg_)
                self.dPf += self.cs[t-1] * dfg_
                dct_f  = self.Pf * dfg_
                self.dbf += dfg_

                dig_ = self.i_gate[t]*(1-self.i_gate[t])*dig
                self.dWi += np.dot(dig_,x.T)
                self.dRi += np.dot(dig_,self.hs[t-1].T)
                dht_i = np.dot(self.Ri.T,dig_)
                self.dPi += self.cs[t-1]*dig_
                dct_i = self.Pi * dig_
                self.dbi += dig_

            return self.loss,self.dWi,self.dWf,self.dWz,self.dWo,self.dWout,self.dRi,self.dRf,self.dRz,self.dRo,self.dPi,self.dPo,self.dPf,self.dbi,self.dbo,self.dbf,self.dbz,self.dbout,self.hs[self.time_steps-1],self.cs[self.time_steps-1]
        
        if self.mode == 'blstm':
            for t in reversed(range(self.time_steps)):
                x = np.zeros((self.len_of_vocab,1))
                x[self.input[t],0] = 1
                
                self.dWout += np.dot(bdout[t],self.hs[t].T)
                dht = np.dot(self.Wout.T,bdout[t]) + dht_z + dht_f + dht_o + dht_i

                dog = np.tanh(self.cs[t])*dht
                dog_ = self.o_gate[t]*(1-self.o_gate[t])*dog
                self.dWo += np.dot(dog_,x.T)
                self.dRo += np.dot(dog_,self.hs[t-1].T)
                dht_o = np.dot(self.Ro.T,dog_)
                self.dPo += self.cs[t]*dog_
                dct_o = self.Po * dog_
                self.dbo += dog_

                dct = (1-np.tanh(self.cs[t])*np.tanh(self.cs[t]))*self.o_gate[t]*dht + dct_cs + dct_f + dct_o + dct_i
                dig = self.z[t] * dct
                dz  = self.i_gate[t] * dct
                dfg = self.cs[t-1] * dct
                dct_cs = self.f_gate[t] * dct

                dz_ = (1-self.z[t]*self.z[t])*dz
                self.dWz += np.dot(dz_,x.T)
                self.dRz += np.dot(dz_,self.hs[t-1].T)
                dht_z = np.dot(self.Rz.T,dz_)
                self.dbz += dz_

                dfg_ = self.f_gate[t]*(1-self.f_gate[t])*dfg
                self.dWf += np.dot(dfg_,x.T)
                self.dRf += np.dot(dfg_,self.hs[t-1].T)
                dht_f = np.dot(self.Rf.T,dfg_)
                self.dPf += self.cs[t-1] * dfg_
                dct_f  = self.Pf * dfg_
                self.dbf += dfg_

                dig_ = self.i_gate[t]*(1-self.i_gate[t])*dig
                self.dWi += np.dot(dig_,x.T)
                self.dRi += np.dot(dig_,self.hs[t-1].T)
                dht_i = np.dot(self.Ri.T,dig_)
                self.dPi += self.cs[t-1]*dig_
                dct_i = self.Pi * dig_
                self.dbi += dig_

            return self.dWi,self.dWf,self.dWz,self.dWo,self.dWout,self.dRi,self.dRf,self.dRz,self.dRo,self.dPi,self.dPo,self.dPf,self.dbi,self.dbo,self.dbf,self.dbz,self.hs[self.time_steps-1],self.cs[self.time_steps-1]

    