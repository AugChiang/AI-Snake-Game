import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = nn.Dropout(p=0.1)(x)
        x = self.linear_2(x)
        x = nn.Dropout(p=0.1)(x)
        x = self.linear_3(x)
        return x
    
    def save(self, file_name='model.pth'):
        save_dir = './model'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        file_name = os.path.join(save_dir, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.opt = opt.Adam(model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, lose):
        # adjust to tensor so that this function can take
        # either a batch of data or a single piece
        # shape: (batch, x...)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
            # (1, x...)            
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            lose = (lose,)
        
        # update Q value
        # Q = model.predict(state_0)
        # new Q = R + gamma * max(Q(state_1))
        pred = self.model(state) # current state
        # preds[argmax(action)] = new Q value
        preds = pred.clone()
        for idx in range(len(lose)):
            newQ = reward[idx]
            if not lose:
                newQ = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            preds[idx][torch.argmax(action).item()] = newQ
        
        # loss function
        self.opt.zero_grad()
        loss = self.loss_function(preds, pred)
        loss.backward()

        self.opt.step()
        

