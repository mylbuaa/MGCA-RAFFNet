import torch.nn as nn
class crossatt(nn.Module):
    def __init__(self, num_heads, dim,dim2,dim3):
        super().__init__()
        self.q = nn.Linear(dim, dim3) 
        self.k = nn.Linear(dim, dim3)
        self.v = nn.Linear(dim, dim3)
        
        self.q4 = nn.Linear(dim, dim3)
        self.k4 = nn.Linear(dim, dim3)
        self.v4 = nn.Linear(dim, dim3)
        
        self.num_heads = num_heads
        self.c = num_heads//num_heads
        self.drop=nn.Dropout(0.1) 

        # Positional encoding for both input sequences
        self.pe1=PositionalEncoding(dim3, max_len=dim2)
        self.pe2=PositionalEncoding(dim3, max_len=dim2)

        # Layer normalization for both input sequences
        self.layer_norm_x1 = nn.LayerNorm(dim)
        self.layer_norm_x4 = nn.LayerNorm(dim)


	
    def forward(self, x1,x4):
        B1, N1, C1 = x1.shape 
        B4, N4, C4 = x4.shape   

        # Positional encoding
        # x1 = self.pe1(x1)
        # x4 = self.pe2(x4)

        # Layer normalization
        x1 = self.layer_norm_x1(x1)
        x4 = self.layer_norm_x4(x4)
        
        q1 = self.q(x1).reshape(B1, N1, self.num_heads, -1).permute(0, 2, 1, 3) 
        q1=self.drop(q1) 
        # print(q1.shape)
        k1 = self.k(x1).reshape(B1, N1, self.num_heads, -1).permute(0, 2, 1, 3)
        k1=self.drop(k1)
        v1 = self.v(x1).reshape(B1, N1, self.num_heads, -1).permute(0, 2, 1, 3)
        v1=self.drop(v1)  

        
        q4 = self.q4(x4).reshape(B4, N4, self.num_heads, -1).permute(0, 2, 1, 3)
        k4 = self.k4(x4).reshape(B4, N4, self.num_heads, -1).permute(0, 2, 1, 3)
        v4 = self.k4(x4).reshape(B4, N4, self.num_heads, -1).permute(0, 2, 1, 3)
        qk1=q1 * k1 
        qk1=torch.softmax(qk1,dim=-1) 
        qk4=q4*k4 
        qk4=torch.softmax(qk4,dim=-1) 
        att1=qk4*v1
        att2=qk1*v4
        att1=att1.reshape(B1,N1,-1)
        att2=att2.reshape(B1,N1,-1)
        att1=torch.add(att1, x1)
        att2=torch.add(att2, x4)
        qk=torch.cat([att1,att2],dim=-1) 
        return qk


  class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
