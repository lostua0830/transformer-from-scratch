import torch
import torch.nn.functional as F
import torch.nn as nn
import math




class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_head):
        super().__init__()
        self.Wo = nn.Linear(d_model,d_model,bias=False)
        self.Wq = nn.Linear(d_model,d_model,bias=False)
        self.Wv = nn.Linear(d_model,d_model,bias=False)
        self.Wk = nn.Linear(d_model,d_model,bias=False)
        self.d_model = d_model
        self.num_head = num_head

        assert d_model %num_head == 0
    
    def forward(self,x,memory=None,causal=False,pad_mask=None,past_k=None,past_v=None,use_cache=False,memory_k=None,memory_v=None):
        Q = self.split_heads(self.Wq(x))
        if memory is None:
            K = self.split_heads(self.Wk(x))
            V = self.split_heads(self.Wv(x))
            if past_k is not None and past_v is not None:
                K = torch.cat([past_k,K],dim=2)
                V = torch.cat([past_v,V],dim=2)
        else:
            if memory_k is not None and memory_v is not None:
                K,V = memory_k,memory_v
            else:
                K = self.split_heads(self.Wk(memory))
                V = self.split_heads(self.Wv(memory))
        attention = self.scaled_dot_product_attention(Q,K,V,causal,pad_mask)
        out = self.merge_heads(attention)
        out = self.Wo(out)
        if use_cache:
            return out,K,V
        return out
    
    def scaled_dot_product_attention(self,Q,K,V,causal=False,pad_mask=None):
        dk = Q.size(-1)
        score = Q @ K.transpose(-2,-1)/math.sqrt(dk)
        if causal:
            Tq = score.size(-2)
            Tk = score.size(-1)
            if Tq == Tk:
                mask = torch.triu(
                    torch.ones(Tq, Tk, device=score.device, dtype=torch.bool),
                    diagonal=1
                )
                score = score.masked_fill(mask, float("-inf"))
        if pad_mask is not None:
            score = self.apply_pad_mask(score,pad_mask)
        score = F.softmax(score,dim=-1)
        attention = score @ V
        return attention

    def apply_pad_mask(self,scores,pad_mask):
        pad_mask = pad_mask[:, None, None, :]  
        scores = scores.masked_fill(pad_mask, float("-inf"))
        return scores
    
    def split_heads(self,x):
        B,T,_ = x.shape
        H = self.num_head
        dk = self.d_model//H
        x= x.contiguous().view(B,T,H,dk)
        x = x.transpose(1,2)
        return x
    
    def merge_heads(self,x):
        x = x.transpose(1,2)
        B,T,H,dk = x.shape
        x = x.contiguous().view(B,T,H*dk)
        return x
