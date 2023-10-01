from model import Actor
from model import Critic
import torch
import torch.nn.functional as F
from replaybuffer import ReplayBuffer
import pybullet_envs


class TD3(object):

    def __init__(self, state_dim, action_dim, max_action, device=None):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

        self.critic.load_the_model()
        self.critic_target.load_the_model()
        self.actor.load_the_model()
        self.actor_target.load_the_model()

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer, epochs, batch_size=100, discount=0.99, tau=0.005,
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for epoch in range(epochs):
            if replay_buffer.can_sample(batch_size):
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                    batch_size=100)

                state = torch.Tensor(batch_states).to(self.device)
                next_state = torch.Tensor(batch_next_states).to(self.device)
                action = torch.Tensor(batch_actions).to(self.device)
                reward = torch.Tensor(batch_rewards).to(self.device)
                done = torch.Tensor(batch_dones).to(self.device)

                # Step 5: From the next state s', the actor target plays the next action a'
                next_action = self.actor_target(next_state).to(self.device)

                # Step 6: Add Gaussian noise
                noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
                noise = noise.clamp(-noise_clip, +noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # Step 7: Get critic q value
                target_q1, target_q2 = self.critic_target(next_state, next_action)

                # Step 8: We keep the minimum of these two Q-values
                target_q = torch.min(target_q1, target_q2)

                # Step 9: We get the final target of the two Critic models, which is Qt = r + y * min(Qt1, Qt2), where y is the discount factor.
                target_q = reward + ((1 - done) * discount * target_q).detach()

                # Step 10: The two critic models should take each the couple (s, a) as input and return two Q-Values(Q1 of s,a and Q2 of s,a)
                current_q1, current_q2 = self.critic(state, action)

                # Step 11
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                # Step 12: Compute the loss between the two critic models: Critic Loss = MSE_Loss(Q(s,a), Qt) + MSE_Loss(Q(s,a), Qt
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Step 13: Once every two iterations, update the actor model by performing gradient ascent on the output of the first critic model.
                if epoch % policy_freq == 0:
                    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Step 14: Still once every two iterations, use Polyak averaging to update the target weights
                    for target_param, main_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

                    for target_param, main_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

            # Making a save method to save a trained model

    def save(self, filename, directory):
        self.actor.save_the_model()