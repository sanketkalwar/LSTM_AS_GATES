import numpy as np
from lstm import LSTM
from dataloader import idx_to_char,char_to_idx,dataset,len_of_dataset,len_of_vocab
from utils import sigmoid,softmax 
import matplotlib.pyplot as plt

plt.ion()

#hyperparameter initialization
lr = 1e-1
time_steps = 100
start_ptr = 0
mean = 0.
std = 0.01
epoches = 50000

x=[]
y=[]
n = 0

bout = np.zeros((len_of_vocab,1))
Wout = np.random.normal(mean,std,(len_of_vocab,len_of_vocab))
mbout = np.zeros_like(bout)
mWout = np.zeros_like(Wout)

lstm_ig  = LSTM(lr=lr,time_steps=time_steps,len_of_vocab=len_of_vocab,mean=mean,std=std,mode='blstm')
lstm_og  = LSTM(lr=lr,time_steps=time_steps,len_of_vocab=len_of_vocab,mean=mean,std=std,mode='blstm')
lstm_fg  = LSTM(lr=lr,time_steps=time_steps,len_of_vocab=len_of_vocab,mean=mean,std=std,mode='blstm')
lstm_im  = LSTM(lr=lr,time_steps=time_steps,len_of_vocab=len_of_vocab,mean=mean,std=std,mode='blstm')


mWi1,mWf1,mWz1,mWo1,mWout1 = np.zeros_like(lstm_ig.Wi),np.zeros_like(lstm_ig.Wf),np.zeros_like(lstm_ig.Wz),np.zeros_like(lstm_ig.Wo),np.zeros_like(lstm_ig.Wout)
mRi1,mRf1,mRz1,mRo1 = np.zeros_like(lstm_ig.Ri),np.zeros_like(lstm_ig.Rf),np.zeros_like(lstm_ig.Rz),np.zeros_like(lstm_ig.Ro)
mPi1,mPo1,mPf1 = np.zeros_like(lstm_ig.Pi),np.zeros_like(lstm_ig.Po),np.zeros_like(lstm_ig.Pf)
mbi1,mbo1,mbf1,mbz1 = np.zeros_like(lstm_ig.bi),np.zeros_like(lstm_ig.bo),np.zeros_like(lstm_ig.bf),np.zeros_like(lstm_ig.bz)

mWi2,mWf2,mWz2,mWo2,mWout2 = np.zeros_like(lstm_og.Wi),np.zeros_like(lstm_og.Wf),np.zeros_like(lstm_og.Wz),np.zeros_like(lstm_og.Wo),np.zeros_like(lstm_og.Wout)
mRi2,mRf2,mRz2,mRo2 = np.zeros_like(lstm_og.Ri),np.zeros_like(lstm_og.Rf),np.zeros_like(lstm_og.Rz),np.zeros_like(lstm_og.Ro)
mPi2,mPo2,mPf2 = np.zeros_like(lstm_og.Pi),np.zeros_like(lstm_og.Po),np.zeros_like(lstm_og.Pf)
mbi2,mbo2,mbf2,mbz2 = np.zeros_like(lstm_og.bi),np.zeros_like(lstm_og.bo),np.zeros_like(lstm_og.bf),np.zeros_like(lstm_og.bz)

mWi3,mWf3,mWz3,mWo3,mWout3 = np.zeros_like(lstm_fg.Wi),np.zeros_like(lstm_fg.Wf),np.zeros_like(lstm_fg.Wz),np.zeros_like(lstm_fg.Wo),np.zeros_like(lstm_fg.Wout)
mRi3,mRf3,mRz3,mRo3 = np.zeros_like(lstm_fg.Ri),np.zeros_like(lstm_fg.Rf),np.zeros_like(lstm_fg.Rz),np.zeros_like(lstm_fg.Ro)
mPi3,mPo3,mPf3 = np.zeros_like(lstm_fg.Pi),np.zeros_like(lstm_fg.Po),np.zeros_like(lstm_fg.Pf)
mbi3,mbo3,mbf3,mbz3 = np.zeros_like(lstm_fg.bi),np.zeros_like(lstm_fg.bo),np.zeros_like(lstm_fg.bf),np.zeros_like(lstm_fg.bz)

mWi4,mWf4,mWz4,mWo4,mWout4 = np.zeros_like(lstm_im.Wi),np.zeros_like(lstm_im.Wf),np.zeros_like(lstm_im.Wz),np.zeros_like(lstm_im.Wo),np.zeros_like(lstm_im.Wout)
mRi4,mRf4,mRz4,mRo4 = np.zeros_like(lstm_im.Ri),np.zeros_like(lstm_im.Rf),np.zeros_like(lstm_im.Rz),np.zeros_like(lstm_im.Ro)
mPi4,mPo4,mPf4 = np.zeros_like(lstm_im.Pi),np.zeros_like(lstm_im.Po),np.zeros_like(lstm_im.Pf)
mbi4,mbo4,mbf4,mbz4 = np.zeros_like(lstm_im.bi),np.zeros_like(lstm_im.bo),np.zeros_like(lstm_im.bf),np.zeros_like(lstm_im.bz)


mparams_of_lstm = [mWi1,mWf1,mWz1,mWo1,mWout1,mRi1,mRf1,mRz1,mRo1,mPi1,mPo1,mPf1,mbi1,mbo1,mbf1,mbz1,\
                    mWi2,mWf2,mWz2,mWo2,mWout2,mRi2,mRf2,mRz2,mRo2,mPi2,mPo2,mPf2,mbi2,mbo2,mbf2,mbz2,\
                    mWi3,mWf3,mWz3,mWo3,mWout3,mRi3,mRf3,mRz3,mRo3,mPi3,mPo3,mPf3,mbi3,mbo3,mbf3,mbz3,\
                    mWi4,mWf4,mWz4,mWo4,mWout4,mRi4,mRf4,mRz4,mRo4,mPi4,mPo4,mPf4,mbi4,mbo4,mbf4,mbz4,\
                    mWout,mbout]


h_prev_ig,c_prev_ig = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))
h_prev_og,c_prev_og = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))
h_prev_fg,c_prev_fg = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))
h_prev_im,c_prev_im = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))



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


def explstm_forward(h_prev,c_prev,ig,im,og,fg,output):
	lig = {}
	lim = {}
	lfg = {}
	log = {} 
	cs  = {}
	cs[-1] = np.copy(c_prev)
	hs  = {}
	p = {}
	loss = 0.
	for t in range(time_steps):
		i = np.dot(ig.Wout,ig.hs[t])+ig.bout
		lig[t] = sigmoid(i)

		m = np.dot(im.Wout,im.hs[t])+im.bout
		lim[t] = np.tanh(m)

		f = np.dot(fg.Wout,fg.hs[t])+fg.bout
		lfg[t] = sigmoid(f)

		o = np.dot(og.Wout,og.hs[t])+og.bout
		log[t] = sigmoid(o)

		cs[t] = lfg[t] * cs[t-1] + lig[t] * lim[t]

		hs[t] = log[t] * np.tanh(cs[t])

		y = np.dot(Wout,hs[t])+bout

		p[t] = softmax(y)

		loss += -np.log(p[t][output[t],0])



	dWout =np.zeros_like(Wout)
	dbout = np.zeros_like(bout)
	dig_out = {}
	dim_out = {}
	dot_out = {}
	dft_out = {}
	dcs_prev_lfg = np.zeros((len_of_vocab,1))
	for t in reversed(range(time_steps)):
		dy = np.copy(p[t])
		dy[output[t],0] -= 1

		dWout += np.dot(dy,hs[t].T)
		dhs  = np.dot(Wout.T,dy)
		dbout += dy

		dlog = np.tanh(cs[t])*dhs 
		dcs = log[t]*(1-np.tanh(cs[t])*np.tanh(cs[t]))*dhs+dcs_prev_lfg

		dlfg = cs[t-1]*dcs

		dcs_prev_lfg = lfg[t]*dcs

		dlig = lim[t]*dcs
		dlim = lig[t]*dcs

		dot = log[t]*(1-log[t])*dlog
		dft = lfg[t]*(1-lfg[t])*dlfg
		dim = (1-lim[t]*lim[t])*dlim
		dit = lig[t]*(1-lig[t])*dlig

		dig_out[t] = np.dot(ig.Wout.T,dit)
		dim_out[t] = np.dot(im.Wout.T,dim)
		dot_out[t] = np.dot(og.Wout.T,dot)
		dft_out[t] = np.dot(fg.Wout.T,dft)
	return loss,hs[time_steps-1],cs[time_steps-1],dig_out,dim_out,dft_out,dot_out,dWout,dbout




    

    


h_prev_exp = np.zeros((len_of_vocab,1))
c_prev_exp = np.zeros((len_of_vocab,1))
smooth_loss = -np.log(1/len_of_vocab)*time_steps
while True:
    if start_ptr+time_steps>len_of_dataset:
        start_ptr = 0
    else:
        input = [char_to_idx[c] for c in dataset[start_ptr:start_ptr+time_steps]]
        output = [char_to_idx[c] for c in dataset[start_ptr+1:start_ptr+time_steps+1]]
        lstm_ig.zero_grad()
        lstm_ig.set_input(input=input,output=output)
        lstm_ig.forward(h_prev=h_prev_exp,c_prev=c_prev_ig)
        c_prev_ig = np.copy(lstm_ig.cs[time_steps-1])
        
        lstm_im.zero_grad()
        lstm_im.set_input(input=input,output=output)
        lstm_im.forward(h_prev=h_prev_exp,c_prev=c_prev_im)
        c_prev_im = np.copy(lstm_im.cs[time_steps-1])

        lstm_fg.zero_grad()
        lstm_fg.set_input(input=input,output=output)
        lstm_fg.forward(h_prev=h_prev_exp,c_prev=c_prev_fg)
        c_prev_fg = np.copy(lstm_fg.cs[time_steps-1])

        lstm_og.zero_grad()
        lstm_og.set_input(input=input,output=output)
        lstm_og.forward(h_prev=h_prev_exp,c_prev=c_prev_og)
        c_prev_og = np.copy(lstm_og.cs[time_steps-1])

        loss,h_prev_exp,c_prev_exp,dig_out,dim_out,dft_out,dot_out,dWout,dbout=explstm_forward(h_prev=h_prev_exp,c_prev=c_prev_exp,ig=lstm_ig,im=lstm_im,og=lstm_og,fg=lstm_fg,output=output)
        lstm_ig.backward(bdout=dig_out)
        lstm_im.backward(bdout=dim_out)
        lstm_fg.backward(bdout=dft_out)
        lstm_og.backward(bdout=dot_out)

        params_of_lstm = [lstm_ig.Wi,lstm_ig.Wf,lstm_ig.Wz,lstm_ig.Wo,lstm_ig.Wout,\
            lstm_ig.Ri,lstm_ig.Rf,lstm_ig.Rz,lstm_ig.Ro,lstm_ig.Pi,lstm_ig.Po,\
            lstm_ig.Pf,lstm_ig.bi,lstm_ig.bo,lstm_ig.bf,lstm_ig.bz,\
            lstm_og.Wi,lstm_og.Wf,lstm_og.Wz,lstm_og.Wo,lstm_og.Wout,\
            lstm_og.Ri,lstm_og.Rf,lstm_og.Rz,lstm_og.Ro,lstm_og.Pi,lstm_og.Po,\
            lstm_og.Pf,lstm_og.bi,lstm_og.bo,lstm_og.bf,lstm_og.bz,\
            lstm_fg.Wi,lstm_fg.Wf,lstm_fg.Wz,lstm_fg.Wo,lstm_fg.Wout,\
            lstm_fg.Ri,lstm_fg.Rf,lstm_fg.Rz,lstm_fg.Ro,lstm_fg.Pi,lstm_fg.Po,\
            lstm_fg.Pf,lstm_fg.bi,lstm_fg.bo,lstm_fg.bf,lstm_fg.bz,\
            lstm_im.Wi,lstm_im.Wf,lstm_im.Wz,lstm_im.Wo,lstm_im.Wout,\
            lstm_im.Ri,lstm_im.Rf,lstm_im.Rz,lstm_im.Ro,lstm_im.Pi,lstm_im.Po,\
            lstm_im.Pf,lstm_im.bi,lstm_im.bo,lstm_im.bf,lstm_im.bz,\
            Wout,bout]

        dparams_of_lstm = [lstm_ig.dWi,lstm_ig.dWf,lstm_ig.dWz,lstm_ig.dWo,lstm_ig.dWout,\
            lstm_ig.dRi,lstm_ig.dRf,lstm_ig.dRz,lstm_ig.dRo,lstm_ig.dPi,lstm_ig.dPo,\
            lstm_ig.dPf,lstm_ig.dbi,lstm_ig.dbo,lstm_ig.dbf,lstm_ig.dbz,\
            lstm_og.dWi,lstm_og.dWf,lstm_og.dWz,lstm_og.dWo,lstm_og.dWout,\
            lstm_og.dRi,lstm_og.dRf,lstm_og.dRz,lstm_og.dRo,lstm_og.dPi,lstm_og.dPo,\
            lstm_og.dPf,lstm_og.dbi,lstm_og.dbo,lstm_og.dbf,lstm_og.dbz,\
            lstm_fg.dWi,lstm_fg.dWf,lstm_fg.dWz,lstm_fg.dWo,lstm_fg.dWout,\
            lstm_fg.dRi,lstm_fg.dRf,lstm_fg.dRz,lstm_fg.dRo,lstm_fg.dPi,lstm_fg.dPo,\
            lstm_fg.dPf,lstm_fg.dbi,lstm_fg.dbo,lstm_fg.dbf,lstm_fg.dbz,\
            lstm_im.dWi,lstm_im.dWf,lstm_im.dWz,lstm_im.dWo,lstm_im.dWout,\
            lstm_im.dRi,lstm_im.dRf,lstm_im.dRz,lstm_im.dRo,lstm_im.dPi,lstm_im.dPo,\
            lstm_im.dPf,lstm_im.dbi,lstm_im.dbo,lstm_im.dbf,lstm_im.dbz,\
            dWout,dbout]
        
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
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.plot(x,y,color='r')
            plt.pause(1e-9)
            plt.savefig('blstm.png')

    start_ptr += time_steps
    n+=1