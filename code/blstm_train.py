import numpy as np
from lstm import LSTM
from dataloader import idx_to_char,char_to_idx,dataset,len_of_dataset,len_of_vocab
from utils import sigmoid,softmax 
import matplotlib.pyplot as plt

plt.ion()

#hyperparameter initialization
lr = 1e-1
time_steps = 25
start_ptr = 0
mean = 0.
std = 0.01
epoches = 50000

x=[]
y=[]
n = 0

bout = np.zeros((len_of_vocab,1))
mbout = np.zeros_like(bout)

lstm_layer1 = LSTM(lr=lr,time_steps=time_steps,len_of_vocab=len_of_vocab,mean=mean,std=std,mode='blstm')
lstm_layer2 = LSTM(lr=lr,time_steps=time_steps,len_of_vocab=len_of_vocab,mean=mean,std=std,mode='blstm')

mWi1,mWf1,mWz1,mWo1,mWout1 = np.zeros_like(lstm_layer1.Wi),np.zeros_like(lstm_layer1.Wf),np.zeros_like(lstm_layer1.Wz),np.zeros_like(lstm_layer1.Wo),np.zeros_like(lstm_layer1.Wout)
mRi1,mRf1,mRz1,mRo1 = np.zeros_like(lstm_layer1.Ri),np.zeros_like(lstm_layer1.Rf),np.zeros_like(lstm_layer1.Rz),np.zeros_like(lstm_layer1.Ro)
mPi1,mPo1,mPf1 = np.zeros_like(lstm_layer1.Pi),np.zeros_like(lstm_layer1.Po),np.zeros_like(lstm_layer1.Pf)
mbi1,mbo1,mbf1,mbz1 = np.zeros_like(lstm_layer1.bi),np.zeros_like(lstm_layer1.bo),np.zeros_like(lstm_layer1.bf),np.zeros_like(lstm_layer1.bz)

mWi2,mWf2,mWz2,mWo2,mWout2 = np.zeros_like(lstm_layer2.Wi),np.zeros_like(lstm_layer2.Wf),np.zeros_like(lstm_layer2.Wz),np.zeros_like(lstm_layer2.Wo),np.zeros_like(lstm_layer2.Wout)
mRi2,mRf2,mRz2,mRo2 = np.zeros_like(lstm_layer2.Ri),np.zeros_like(lstm_layer2.Rf),np.zeros_like(lstm_layer2.Rz),np.zeros_like(lstm_layer2.Ro)
mPi2,mPo2,mPf2 = np.zeros_like(lstm_layer2.Pi),np.zeros_like(lstm_layer2.Po),np.zeros_like(lstm_layer2.Pf)
mbi2,mbo2,mbf2,mbz2 = np.zeros_like(lstm_layer2.bi),np.zeros_like(lstm_layer2.bo),np.zeros_like(lstm_layer2.bf),np.zeros_like(lstm_layer2.bz)


mparams_of_lstm = [mWi1,mWf1,mWz1,mWo1,mWout1,mRi1,mRf1,mRz1,mRo1,mPi1,mPo1,mPf1,mbi1,mbo1,mbf1,mbz1,\
                    mWi2,mWf2,mWz2,mWo2,mWout2,mRi2,mRf2,mRz2,mRo2,mPi2,mPo2,mPf2,mbi2,mbo2,mbf2,mbz2,mbout]

smooth_loss = -np.log(1/len_of_vocab)*time_steps
h_prev1,c_prev1 = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))
h_prev2,c_prev2 = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))


def blstm_backward(h1,w1,h2,w2,output):
    p={}
    dout = {}
    loss = 0
    for t in reversed(range(time_steps)):
        out = np.dot(w1,h1[t])+np.dot(w2,h2[time_steps-t-1])+bout
        p[t]=softmax(out)
        loss += -np.log(p[t][output[t],0])
    dbout = np.zeros((len_of_vocab,1))
    for t in reversed(range(time_steps)):
        dout[t]=np.copy(p[t])
        dout[t][output[t],0] -= 1
        dbout += dout[t]

    return loss,dout,dbout


    

def forward(h_p,c_p,num_char,lstm,x):
    hs = np.copy(h_p)
    cs = np.copy(c_p)
    I = np.dot(lstm.Wi,x) + np.dot(lstm.Ri,hs) + lstm.Pi*cs + lstm.bi
    i_gate = sigmoid(I)
    F = np.dot(lstm.Wf,x) + np.dot(lstm.Rf,hs) + lstm.Pf*cs + lstm.bf
    f_gate = sigmoid(F)
    Z = np.dot(lstm.Wz,x) + np.dot(lstm.Rz,hs) + lstm.bz
    z = np.tanh(Z)
    cs = i_gate*z + f_gate*cs
    O = np.dot(lstm.Wo,x) + np.dot(lstm.Ro,hs) + lstm.Po*cs +lstm.bo
    o_gate = sigmoid(O)
    hs = o_gate * np.tanh(cs)
    return hs,cs

def sample(h_p1,c_p1,h_p2,c_p2,num_char):
    hs1,cs1=np.copy(h_p1),np.copy(c_p1)
    hs2,cs2=np.copy(h_p2),np.copy(c_p2)
    x = np.zeros((len_of_vocab,1))
    x[np.random.randint(0,len_of_vocab),0] = 1
    idxs = []
    for _ in range(num_char):
        hs1,cs1=forward(hs1,cs1,num_char,lstm=lstm_layer1,x=x)
        hs2,cs2=forward(hs2,cs2,num_char,lstm=lstm_layer2,x=x)
        out = np.dot(lstm_layer1.Wout,hs1)+np.dot(lstm_layer2.Wout,hs2)+bout
        p=softmax(out)
        idx = np.random.choice(len_of_vocab,1,p=p.ravel())[0]
        x = np.zeros((len_of_vocab,1))
        x[idx,0] = 1
        idxs.append(idx)
    print('--------------------------------------------------------------')
    print(''.join(idx_to_char[c] for c in idxs ))
    print('--------------------------------------------------------------')

    

    



smooth_loss = -np.log(1/len_of_vocab)*time_steps
while n<=epoches:
    if start_ptr+time_steps>len_of_dataset:
        start_ptr = 0
        # h_prev1 = np.zeros((len_of_vocab,1))
        # h_prev2 = np.zeros((len_of_vocab,1))
    else:
        input = [char_to_idx[c] for c in dataset[start_ptr:start_ptr+time_steps]]
        output = [char_to_idx[c] for c in dataset[start_ptr+1:start_ptr+time_steps+1]]
        lstm_layer1.zero_grad()
        lstm_layer1.set_input(input=input,output=output)
        lstm_layer1.forward(h_prev=h_prev1,c_prev=c_prev1)
        
        lstm_layer2.zero_grad()
        lstm_layer2.set_input(input=input[::-1],output=output[::-1])
        lstm_layer2.forward(h_prev=h_prev2,c_prev=c_prev2)

        loss,dout,dbout=blstm_backward(h1=lstm_layer1.hs,w1=lstm_layer1.Wout,h2=lstm_layer2.hs,w2=lstm_layer2.Wout,output=output)

        do = [v for v in dout.values()]
        dWi1,dWf1,dWz1,dWo1,dWout1,dRi1,dRf1,dRz1,dRo1,dPi1,dPo1,dPf1,dbi1,dbo1,dbf1,dbz1,h_prev1,c_prev1=lstm_layer1.backward(bdout=do)
        dWi2,dWf2,dWz2,dWo2,dWout2,dRi2,dRf2,dRz2,dRo2,dPi2,dPo2,dPf2,dbi2,dbo2,dbf2,dbz2,h_prev2,c_prev2=lstm_layer2.backward(bdout=do)
        for dparam in[dWi1,dWf1,dWz1,dWo1,dWout1,dRi1,dRf1,dRz1,dRo1,dPi1,dPo1,dPf1,dbi1,dbo1,dbf1,dbz1,\
            dWi2,dWf2,dWz2,dWo2,dWout2,dRi2,dRf2,dRz2,dRo2,dPi2,dPo2,dPf2,dbi2,dbo2,dbf2,dbz2,dbout] :
            np.clip(dparam,-1,1,out=dparam)
        params_of_lstm = [lstm_layer1.Wi,lstm_layer1.Wf,lstm_layer1.Wz,lstm_layer1.Wo,lstm_layer1.Wout,\
            lstm_layer1.Ri,lstm_layer1.Rf,lstm_layer1.Rz,lstm_layer1.Ro,lstm_layer1.Pi,lstm_layer1.Po,\
            lstm_layer1.Pf,lstm_layer1.bi,lstm_layer1.bo,lstm_layer1.bf,lstm_layer1.bz,\
            lstm_layer2.Wi,lstm_layer2.Wf,lstm_layer2.Wz,lstm_layer2.Wo,lstm_layer2.Wout,\
            lstm_layer2.Ri,lstm_layer2.Rf,lstm_layer2.Rz,lstm_layer2.Ro,lstm_layer2.Pi,lstm_layer2.Po,\
            lstm_layer2.Pf,lstm_layer2.bi,lstm_layer2.bo,lstm_layer2.bf,lstm_layer2.bz,bout]
        dparams_of_lstm = [dWi1,dWf1,dWz1,dWo1,dWout1,dRi1,dRf1,dRz1,dRo1,dPi1,dPo1,dPf1,dbi1,dbo1,dbf1,dbz1,\
            dWi2,dWf2,dWz2,dWo2,dWout2,dRi2,dRf2,dRz2,dRo2,dPi2,dPo2,dPf2,dbi2,dbo2,dbf2,dbz2,dbout]
        
        for params,dparams,mparams in zip(params_of_lstm,dparams_of_lstm,mparams_of_lstm):
            mparams += dparams*dparams
            params +=-lr*dparams/np.sqrt(mparams+1e-8)

        smooth_loss = (0.999*smooth_loss)+(0.001*loss)
        x.append(n)
        y.append(smooth_loss)
        if n % 1000 == 0:
            print('--------------------------------------------')
            print('iter:',n)
            print('smooth_loss:',smooth_loss)
            sample(h_p1=h_prev1,c_p1=c_prev1,h_p2=h_prev2,c_p2=c_prev2,num_char=300)
            print('--------------------------------------------')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.plot(x,y,color='r')
            plt.pause(1e-9)

    start_ptr += time_steps
    n+=1
plt.savefig('../../Performance/blstm.png')