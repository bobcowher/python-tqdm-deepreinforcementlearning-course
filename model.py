import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        print(f"Initial state dimensions for the actor are: {state_dim}")
        self.layer_1 = nn.Linear(state_dim, 600)
        self.layer_2 = nn.Linear(600, 400)
        self.layer_3 = nn.Linear(400, 300)
        self.layer_4 = nn.Linear(300, action_dim)
        self.dropout = nn.Dropout(0.5)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.dropout(x)
        x = F.relu(self.layer_2(x))
        x = self.dropout(x)
        x = F.relu(self.layer_3(x))
        x = self.max_action * torch.tanh(self.layer_4(x))
        return x

    def save_the_model(self, weights_filename='actor_latest.pt'):
        weights_filename = "models/" + weights_filename
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename='actor_latest.pt'):
        weights_filename = "models/" + weights_filename
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
            return True
        except:
            print(f"No weights file available at {weights_filename}")
            return False

    def print_model(self):
        for name, param in self.named_parameters():
            print(name, param.data)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # First Critic Network.
        print(f"Initial state dimensions for the critic are: {state_dim + action_dim}")
        self.layer_1 = nn.Linear(state_dim + action_dim, 600)
        self.layer_2 = nn.Linear(600, 400)
        self.layer_3 = nn.Linear(400, 300)
        self.layer_4 = nn.Linear(300, 1)

        # Second critic network
        self.layer_5 = nn.Linear(state_dim + action_dim, 600)
        self.layer_6 = nn.Linear(600, 400)
        self.layer_7 = nn.Linear(400, 300)
        self.layer_8 = nn.Linear(300, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, u):

        xu = torch.cat([x, u], 1)

        # First critic forward prop
        x1 = F.relu(self.layer_1(xu))
        x1 = self.dropout(x1)
        x1 = F.relu(self.layer_2(x1))
        x1 = self.dropout(x1)
        x1 = F.relu(self.layer_3(x1))
        x1 = self.layer_4(x1)


        # Second critic forward prop
        x2 = F.relu(self.layer_5(xu))
        x2 = self.dropout(x2)
        x2 = F.relu(self.layer_6(x2))
        x2 = self.dropout(x2)
        x2 = F.relu(self.layer_7(x2))
        x2 = self.layer_8(x2)

        return x1, x2

    def Q1(self, x, u):

        xu = torch.cat([x, u], 1)

        # First critic forward prop
        x1 = F.relu(self.layer_1(xu))
        x1 = self.dropout(x1)
        x1 = F.relu(self.layer_2(x1))
        x1 = self.dropout(x1)
        x1 = F.relu(self.layer_3(x1))
        x1 = self.layer_4(x1)

        return x1

    def save_the_model(self, weights_filename='critic_latest.pt'):
        weights_filename = "models/" + weights_filename
        # Take the default weights filename(latest.pt) and save it
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename='critic_latest.pt'):
        weights_filename = "models/" + weights_filename
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
            return True
        except:
            print(f"No weights file available at {weights_filename}")
            return False

    def print_model(self):
        for name, param in self.named_parameters():
            print(name, param.data)

