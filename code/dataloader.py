#load dataset 
dataset = open('../data/input.txt','r').read()
len_of_dataset = len(dataset)
print('length of dataset:',len_of_dataset)

vocab = set(dataset)
len_of_vocab = len(vocab)
print('length of vocab:',len_of_vocab)

char_to_idx = {char:idx for idx,char in enumerate(vocab)}
print('char_to_idx:',char_to_idx)

idx_to_char = {idx:char for idx,char in enumerate(vocab)}
print('idx_to_char:',idx_to_char)