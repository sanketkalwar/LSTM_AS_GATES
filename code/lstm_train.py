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
lstm = LSTM(lr=lr,time_steps=time_steps,len_of_vocab=len_of_vocab,mean=mean,std=std,mode='lstm')
smooth_loss = -np.log(1/len_of_vocab)*time_steps
h_prev,c_prev = np.zeros((len_of_vocab,1)),np.zeros((len_of_vocab,1))
while n<=epoches:
	if start_ptr+time_steps > len_of_dataset:
		start_ptr = 0
		# h_prev = np.zeros((len_of_vocab,1))
	else:
		input = [char_to_idx[c] for c in dataset[start_ptr:start_ptr+time_steps]]
		output = [char_to_idx[c] for c in dataset[start_ptr+1:start_ptr+time_steps+1]]

		lstm.zero_grad()
		lstm.set_input(input=input,output=output)
		lstm.forward(h_prev=h_prev,c_prev=c_prev)
		loss,dWi,dWf,dWz,dWo,dWout,dRi,dRf,dRz,dRo,dPi,dPo,dPf,dbi,dbo,dbf,dbz,dbout,h_prev,c_prev=lstm.backward()
		
		lstm.clip_grad(clip_val=1)
		lstm.step()
		smooth_loss = (0.999*smooth_loss)+(0.001*loss)
		x.append(n)
		y.append(smooth_loss)
		if n % 1000 == 0:
			print('--------------------------------------------')
			print('iter:',n)
			print('smooth_loss:',smooth_loss)
			lstm.sample(h_prev=h_prev,c_prev=c_prev,num_char=300)
			print('--------------------------------------------')
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			plt.plot(x,y,color='r')
			plt.pause(0.000000001)
	n += 1
	start_ptr += time_steps

plt.savefig('../../Performance/lstm_with_peephole_connection.png')