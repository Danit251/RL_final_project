import itertools

import gym
import numpy as np
import torch
from algorithm import Algo
from buffer import ReplayBuffer
from models import DQNModel
from utils import OrnsteinUhlenbeckActionNoise, hard_update

MAX_REPLAY_BUFFER = 100000

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


class DQN(Algo):

    def __init__(self,   env, action_space: dict, buffer, device, noise, apply_noise, discount_factor=0.99, update_rate=50, batch_size=64,
                 lr=0.001, epsilon=1, eps_end=0.01, eps_dec=5e-4, max_steps=1000, velocity=None):
        super().__init__(env, 'dqn')
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = action_space
        self.num_actions = len(self.action_space)
        self.device = device
        self.update_rate = update_rate
        self.batch_size = batch_size
        self.buffer = buffer
        self.discount_factor = discount_factor
        self.max_steps = max_steps
        self.noise = noise
        self.apply_noise = apply_noise
        self.velocity = velocity

        self.step_counter = 0
        self.scores = []

        if self.velocity:
            self.state_dim = env.observation_space.shape[0] - 1
        else:
            self.state_dim = env.observation_space.shape[0]

        self.Q_net = DQNModel(state_dim=self.state_dim, action_dim=self.num_actions, hidden_dim=256).to(self.device)
        self.Q_target = DQNModel(state_dim=self.state_dim, action_dim=self.num_actions, hidden_dim=256).to(self.device)
        self.optimizer = torch.optim.AdamW(self.Q_net.parameters(), self.lr)
        self.loss = torch.nn.HuberLoss()

    def policy(self, observation):
        if np.random.random() < self.epsilon:
            actions = list(self.action_space.keys())
            action = np.random.choice(actions)
            return action
        else:
            state = torch.tensor([observation], dtype=torch.float).to(self.device)
            actions = self.Q_net(state).detach()
            action = torch.argmax(actions, dim=1).cpu().data.numpy()[0]
            return action

    def update_state_for_velocity(self, state):
        velocity_x = state[2]
        velocity_y = state[3]

        if self.velocity == "diff":
            new_velocity = velocity_x - velocity_y
            state = np.concatenate((state[:2], [new_velocity], state[4:]), axis=0)

        if self.velocity == "concat":
            new_velocity = velocity_x + velocity_y
            state = np.concatenate((state[:2], [new_velocity], state[4:]), axis=0)

        if self.velocity == "avg":
            new_velocity = (velocity_x + velocity_y)/2
            state = np.concatenate((state[:2], [new_velocity], state[4:]), axis=0)

        return state

    def run_algo_step(self, i):
        print(f'EPISODE: {i}')
        state = self.env.reset()
        state = self.update_state_for_velocity(state)
        total_reward = 0

        for _ in range(self.max_steps):
            self.env.render()
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(self.action_space[action])
            next_state = state + self.noise.sample().numpy() if self.apply_noise else next_state
            next_state = self.update_state_for_velocity(next_state)
            total_reward += reward
            self.buffer.add(state, action, reward, next_state, done)
            state = next_state
            if self.buffer.len < self.batch_size:
                continue
            if self.step_counter % self.update_rate == 0:
                hard_update(self.Q_target, self.Q_net)

            state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.buffer.sample(self.batch_size)

            # step
            state_batch = state_batch.to(self.device)
            new_state_batch = new_state_batch.to(self.device)
            q_predicted = self.Q_net(state_batch)
            q_predicted = q_predicted.gather(1, action_batch.unsqueeze(1).type(torch.int64))
            q_next = self.Q_target(new_state_batch).detach()
            q_target = reward_batch + self.discount_factor * q_next.max(1)[0] * (1 - done_batch)

            # train
            self.optimizer.zero_grad()
            self.Q_net.zero_grad()
            out_loss = self.loss(q_predicted, q_target.unsqueeze(1))
            out_loss.backward()
            self.optimizer.step()

            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
            self.step_counter += 1
            if done:
                break

        return total_reward


def main(velocity=None):
    seed = 42

    action_space = {i: v for i, v in enumerate(itertools.product([0, 0.3, 0.6, 0.9], [0, 0.6, -0.6, 0.9, -0.9]))}
    env = gym.make("LunarLanderContinuous-v2")
    env.action_space.np_random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise = OrnsteinUhlenbeckActionNoise(action_dim=1, rng=rng, theta=0.005, sigma=0.005)
    buffer = ReplayBuffer(MAX_REPLAY_BUFFER, device, rng)
    algo = DQN(env, action_space, buffer, device, noise, apply_noise=False, velocity=velocity)
    algo.run_all_episodes()


if __name__ == '__main__':
    main()
