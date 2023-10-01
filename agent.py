from model import Actor
from model import Critic
import torch
import torch.nn.functional as F
from replaybuffer import ReplayBuffer
import pybullet_envs
import numpy as np
from plot import LivePlot
import time


class TD3(object):

    def __init__(self, state_dim, action_dim, max_action, batch_size, policy_freq, discount, tau=0.005, eval_freq=5e3,
                 policy_noise=0.2, expl_noise=0.1, noise_clip=0.5, start_timesteps = 1e4, device=None):
        """

        :param state_dim:
        :param action_dim:
        :param max_action:
        :param batch_size: Size of the sample batch used
        :param policy_freq: Number of iterations to wait before the policy network (Actor model) is updated
        :param device: Device the model should train on (cpu v.s gpu)
        :param discount: The discount factor for future rewards
        :param eval_freq: How often we should evaluate the model
        :param tau: target model update rate
        :param start_timesteps: Number of timesteps to take in warmup mode
        """
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
        self.batch_size = batch_size
        self.policy_freq = policy_freq
        self.discount = discount
        self.eval_freq = eval_freq
        self.tau = tau
        self.policy_noise = policy_noise
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.start_timesteps = start_timesteps

        print(f"Configured agent with device: {self.device}")

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def evaluate_policy(self, env, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = self.select_action(np.array(obs))
                obs, reward, done, _ = env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print("---------------------------------------")
        return avg_reward

    def train(self, env, max_timesteps):
        stats = {'Returns': [], 'AvgReturns': []}

        replay_buffer = ReplayBuffer()

        evaluations = [self.evaluate_policy(env)]

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True
        t0 = time.time()

        plotter = LivePlot()

        while total_timesteps < max_timesteps:

            # If the episode is done
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,
                                                                                  episode_reward))
                    self.learn(replay_buffer=replay_buffer, epochs=episode_timesteps)
                    stats['Returns'].append(episode_reward)

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    evaluations.append(self.evaluate_policy(env))
                    self.save()

                # When the training step is done, we reset the state of the environment
                obs = env.reset()

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if episode_num % 10 == 0:
                    average_returns = np.mean(stats['Returns'][-100:])
                    stats['AvgReturns'].append(average_returns)
                    plotter.update_plot(stats)

            # Before 10000 timesteps, we play random actions
            if total_timesteps < self.start_timesteps:
                action = env.action_space.sample()
            else:  # After 10000 timesteps, we switch to the model
                action = self.select_action(np.array(obs))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if self.expl_noise != 0:
                    action = (action + np.random.normal(0, self.expl_noise, size=env.action_space.shape[0])).clip(
                        env.action_space.low, env.action_space.high)

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done, _ = env.step(action)

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

    def test(self, env, max_timesteps):

        total_timesteps = 0
        episode_num = 0
        done = True
        t0 = time.time()

        plotter = LivePlot()

        while total_timesteps < max_timesteps:
            time.sleep(0.001) # Slow down enough to see the environment run.
            # If the episode is done
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,
                                                                                  episode_reward))

                # When the training step is done, we reset the state of the environment
                obs = env.reset()

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1


            action = self.select_action(np.array(obs))
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if self.expl_noise != 0:
                action = (action + np.random.normal(0, self.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done, _ = env.step(action)

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward


            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1

    def learn(self, replay_buffer: ReplayBuffer, epochs):

        for epoch in range(epochs):
            if replay_buffer.can_sample(self.batch_size):
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size=self.batch_size)

                state = torch.Tensor(batch_states).to(self.device)
                next_state = torch.Tensor(batch_next_states).to(self.device)
                action = torch.Tensor(batch_actions).to(self.device)
                reward = torch.Tensor(batch_rewards).to(self.device)
                done = torch.Tensor(batch_dones).to(self.device)

                # Step 5: From the next state s', the actor target plays the next action a'
                next_action = self.actor_target(next_state).to(self.device)

                # Step 6: Add Gaussian noise
                noise = torch.Tensor(batch_actions).data.normal_(0, self.policy_noise).to(self.device)
                noise = noise.clamp(-self.noise_clip, +self.noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # Step 7: Get critic q value
                target_q1, target_q2 = self.critic_target(next_state, next_action)

                # Step 8: We keep the minimum of these two Q-values
                target_q = torch.min(target_q1, target_q2)

                # Step 9: We get the final target of the two Critic models, which is Qt = r + y * min(Qt1, Qt2), where y is the discount factor.
                target_q = reward + ((1 - done) * self.discount * target_q).detach()

                # Step 10: The two critic models should take each the couple (s, a) as input and return two Q-Values(Q1 of s,a and Q2 of s,a)
                current_q1, current_q2 = self.critic(state, action)

                # Step 11
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                # Step 12: Compute the loss between the two critic models: Critic Loss = MSE_Loss(Q(s,a), Qt) + MSE_Loss(Q(s,a), Qt
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Step 13: Once every two iterations, update the actor model by performing gradient ascent on the output of the first critic model.
                if epoch % self.policy_freq == 0:
                    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Step 14: Still once every two iterations, use Polyak averaging to update the target weights
                    for target_param, main_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

                    for target_param, main_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                        target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

            # Making a save method to save a trained model

    def save(self):
        self.actor.save_the_model()
        self.critic.save_the_model()