# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F

import pdb

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) 
    
class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).cuda()
        item_idx = torch.LongTensor(x[:, 1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()        
    
class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k=4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb).squeeze()
        out = self.sigmoid(h1)
        if is_training:
            return out, z_emb
        else:
            return out    
    
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()      
    
class NCF(nn.Module):

    def __init__(self, num_users, num_items, embedding_k=64):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        out = self.linear_1(z_emb).squeeze()

        if is_training:
            return self.sigmoid(out), z_emb
        else:
            return self.sigmoid(out)

    def fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4, batch_size=128, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = torch.Tensor(y[selected_idx]).cuda()

                optimizer.zero_grad()
                pred = self.forward(sub_x, False)

                xent_loss = nn.MSELoss()(pred, sub_y)
                loss = xent_loss
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def partial_fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4):
        self.fit(x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4)

    def predict(self, x):
        pred, z_emb = self.forward(x, True)
        return pred.detach().cpu().numpy().flatten(), z_emb.detach().cpu().numpy()

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # 1-6960
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                
                xent_loss = self.xent_func(pred,sub_y)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred, u_emb, v_emb = self.forward(x, True)
        z_emb = torch.cat([u_emb, v_emb], axis=1)
        return pred.detach().cpu().numpy(), z_emb.detach().cpu().numpy()
    
class MLP(nn.Module):
    def __init__(self, input_size, *args, **kwargs):
        super().__init__()     
        self.input_size = input_size
        self.linear_1 = torch.nn.Linear(2*self.input_size, 5)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()    
    
    def forward(self, x):        
        
        out = nn.Softmax(dim = 1)(self.linear_1(x).squeeze())
        return out
    
class NCF_ours(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4):
        super(NCF_ours, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.NCF_model = MF_BaseModel(num_users, num_items, embedding_k)
        self.MLP_model = MLP(input_size = self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.propensity_model = NCF_BaseModel(num_users, num_items, embedding_k)
        self.xent_func = torch.nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduction = 'sum')
    def fit(self, x, t, c, y, num_epoch=1000, lr=0.01, lamb=1e-4, tol=1e-4, batch_size=4096, verbose=True,
           alpha1 = 1, alpha = 1, beta = 1, theta = 1, gamma = 1, rho = 1, eta = 1, thr = 0.05):
        
        optimizer_NCF = torch.optim.Adam(self.NCF_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_MLP = torch.optim.Adam(self.MLP_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_prop = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        
        num_sample = len(t)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_t = t[selected_idx]
                sub_c = c[selected_idx]
                sub_y = y[selected_idx]
                
                pred = self.propensity_model.forward(sub_x, False).squeeze()
                pred = torch.clip(pred, thr, 1)

                ctr_loss = nn.MSELoss(reduction = 'sum')(pred, torch.Tensor(sub_t).cuda())

                _, pred_class_emb_u, pred_class_emb_v = self.NCF_model.forward(sub_x, True)
                pred_class_emb = torch.cat([pred_class_emb_u, pred_class_emb_v], axis=1)
                pred_class = self.MLP_model.forward(pred_class_emb)

                L1 = F.binary_cross_entropy(pred * pred_class[:, 0] + 1e-6, torch.Tensor(sub_t * (1-sub_c) * (1-sub_y)).cuda(), reduction = 'sum')
                L2 = F.binary_cross_entropy(pred * pred_class[:, 1] + 1e-6, torch.Tensor(sub_t * (1-sub_c) * sub_y).cuda(), reduction = 'sum')
                L3 = F.binary_cross_entropy(pred * pred_class[:, 2] + 1e-6, torch.Tensor(sub_t * sub_c * (1-sub_y)).cuda(), reduction = 'sum')
                L4 = F.binary_cross_entropy(pred * (pred_class[:, 4] + pred_class[:, 3]) + 1e-6, torch.Tensor(sub_t * sub_c * sub_y).cuda(), reduction = 'sum')
                L5 = F.binary_cross_entropy((1-pred) * (pred_class[:, 4] + pred_class[:, 1]) + 1e-6, torch.Tensor((1-sub_t) * (1-sub_c) * sub_y).cuda(), reduction = 'sum')
                L6 = F.binary_cross_entropy((1-pred) * (pred_class[:, 0] + pred_class[:, 2] + pred_class[:, 3]) + 1e-6, torch.Tensor((1-sub_t) * (1-sub_c) * (1-sub_y)).cuda(), reduction = 'sum')   
                
                loss = (alpha1 * L1 + alpha * L2 + beta * L3 + theta * L4 + gamma * L5 + rho * L6) + eta * ctr_loss
                
                optimizer_NCF.zero_grad()
                optimizer_MLP.zero_grad()
                optimizer_prop.zero_grad()
                loss.backward()                
                optimizer_NCF.step()
                optimizer_MLP.step()
                optimizer_prop.step()
                
                epoch_loss += loss.detach().detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        _, pred_class_emb_u, pred_class_emb_v = self.NCF_model.forward(x, True)
        pred_class_emb = torch.cat([pred_class_emb_u, pred_class_emb_v], axis=1)
        pred_class = self.MLP_model.forward(pred_class_emb)
        return pred_class.detach().cpu().numpy().flatten() 
    
def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)
