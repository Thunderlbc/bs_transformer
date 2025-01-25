import torch
import torch.nn as nn
import numpy as np
import time
import math
import logging
from matplotlib import pyplot
logging.basicConfig(level=logging.INFO)

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)

input_window = 200 # number of input steps
NUM_Bs = 1000
batch_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, d_model],so need seq-len <= 5000
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)
          

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        # 公共的结构
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(feature_size*input_window, NUM_Bs)

        # BA分开的结构
        self.emb_B = nn.Linear(1,feature_size)
        self.emb_A = nn.Linear(1,feature_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,blist, alist):
        # src with shape (input_window, batch_len, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        aemb = self.pos_encoder(self.emb_A(alist))
        print('aemb.shape', aemb.shape)
        bemb = self.pos_encoder(self.emb_B(blist))
        aout = self.transformer_encoder(aemb, self.src_mask)
        bout = self.transformer_encoder(bemb, self.src_mask)
        # seq_len, batch_size, feature_size
        print('aout.shape', aout.shape)

        # seq_len, batch_size, feature_size*2
        allemb = torch.concat((aout,  bout), dim=2)
        print('allemb.shape', allemb.shape)
        allemb = torch.concat
        # 这里我们使用一个全连接层来做预测
        bout = self.decoder(allemb)
        
        
        return bout

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
'''
In fact, assuming that the number of samples is N, 
the length of the input sequence is m, and the backward prediction is k steps, 
then length of a block [input : 1 , 2 ... m  -> output : k , k+1....m+k ] 
should be (m+k) :  block_len, so to ensure that each block is complete, 
the end element of the last block should be the end element of the entire sequence, 
so the actual number of blocks is [N - block_len + 1] 
'''
def get_data(user_ins, train_split=0.7, validate_split=0.2):
    """
    我们预期的输入输出是：
        in: user_id \t Blist[,] ot Alist[,] \t rank \t next_a \t next_b \t next_t
       out: 7:2:1
            train_data: [  ([B0,...,B200], [A0,...,A200], B201)  ]
            validate_data: [  ([B0,...,B200], [A0,...,A200], B201)  ]
            test_data: [  ([B0,...,B200], [A0,...,A200], B201)  ]
        其中，Bi是离散的整数，范围是1~1000; Ai是连续的浮点数，范围是0~2000.0
    """

    # 我们使用user_id hash 来做训练集、验证集、测试集的划o
    import cityhash
    test_split=1-train_split-validate_split
    logging.info("Loading data with train split: {} validate split: {} test split: {}".format(train_split, validate_split, test_split))
    train_data = []
    validate_data = []
    test_data = []
    with open(user_ins, 'r') as fin:
        for line in fin:
            line = line.strip().split("\t")
            user_id,blist,alist,rank,next_b,next_a,next_t  = line
            blist = blist.split(',')
            alist = alist.split(',')
            # 这里我们使用user_id的hash值来做训练集、验证集、测试集的划分
            # 我们使用cityhash来做hash
            hash_val = cityhash.CityHash64(user_id) % 100
            if hash_val < train_split * 100:
                train_data.append((blist,alist,next_b,next_t))
            elif hash_val < (train_split + validate_split) * 100:
                validate_data.append((blist,alist,next_b,next_t))
            else:
                test_data.append((blist,alist,next_b,next_t))

            
    train_tensor = [
        (torch.LongTensor(np.array(blist)).to(device), torch.FloatTensor(np.array(alist)).to(device), torch.FloatTensor(next_b).to(device), torch.FloatTensor(next_t).to(device)) for blist,alist,next_b,next_t in train_data
    ]
    validate_tensor = [
        (torch.LongTensor(np.array(blist)).to(device), torch.FloatTensor(np.array(alist)).to(device), torch.FloatTensor(next_b).to(device), torch.FloatTensor(next_t).to(device)) for blist,alist,next_b,next_t in validate_data
    ]
    test_tensor = [
        (torch.LongTensor(np.array(blist)).to(device), torch.FloatTensor(np.array(alist)).to(device), torch.FloatTensor(next_b).to(device), torch.FloatTensor(next_t).to(device)) for blist,alist,next_b,next_t in test_data
    ]
    return train_tensor, validate_tensor, test_tensor

def get_batch(input_data, i , batch_size):

    # batch_len = min(batch_size, len(input_data) - 1 - i) #  # Now len-1 is not necessary
    batch_len = min(batch_size, len(input_data) - i)
    data = input_data[ i:i + batch_len ]

    # 格式是 blist,alist,nextb,nextt
    # ( seq_len, batch, 1 ) , 1 is feature size
    inblist = torch.stack([item[0] for item in data]).view((input_window,batch_len,1))
    inalist = torch.stack([item[1] for item in data]).view((input_window,batch_len,1))
    # (1, batch, 1)
    target_b = torch.stack([item[2] for item in data]).view((1,batch_len,1))
    target_t = torch.stack([item[3] for item in data]).view((1,batch_len,1))
    return inblist,inalist,target_b,target_t

def train(train_data):
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data), batch_size)):  # Now len-1 is not necessary
        # data and target are the same shape with (input_window,batch_len,1)
        #data, targets = get_batch(train_data, i , batch_size)
        inblist, inalist, target_b, target_t = get_batch(train_data, i , batch_size)
        optimizer.zero_grad()
        output = model(inblist,inalist)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        # for i in range(0, len(data_source) - 1):
        for i in range(len(data_source)):  # Now len-1 is not necessary
            data, target = get_batch(data_source, i , 1) # one-step forecast
            output = eval_model(data)            
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            
    #test_result = test_result.cpu().numpy() -> no need to detach stuff.. 
    len(test_result)

    pyplot.plot(test_result,color="red")
    pyplot.plot(truth[:500],color="blue")
    pyplot.plot(test_result-truth,color="green")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-epoch%d.png'%epoch)
    pyplot.close()
    return total_loss / i


# predict the next n steps based on the input data 
def predict_future(eval_model, data_source,steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    data, _ = get_batch(data_source , 0 , 1)
    with torch.no_grad():
        for i in range(0, steps):            
            output = eval_model(data[-input_window:])
            # (seq-len , batch-size , features-num)
            # input : [ m,m+1,...,m+n ] -> [m+1,...,m+n+1]
            data = torch.cat((data, output[-1:])) # [m,m+1,..., m+n+1]

    data = data.cpu().view(-1)
    
    # I used this plot to visualize if the model pics up any long therm structure within the data.
    pyplot.plot(data,color="red")       
    pyplot.plot(data[:input_window],color="blue")    
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-future%d.png'%steps)
    pyplot.show()
    pyplot.close()
        

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        # for i in range(0, len(data_source) - 1, eval_batch_size): # Now len-1 is not necessary
        for i in range(0, len(data_source), eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

train_data, val_data, test_data = get_data()
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005 
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    if ( epoch % 5 == 0 ):
        val_loss = plot_and_loss(model, val_data,epoch)
        predict_future(model, val_data,200)
    else:
        val_loss = evaluate(model, val_data)
   
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    #if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step() 

#src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number) 
#out = model(src)
#
#print(out)
#print(out.shape)