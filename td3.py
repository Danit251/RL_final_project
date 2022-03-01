import numpy as np
import torch
import gym
import time
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

start_timestep = 1e4
std_noise = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)


# Actor Neural Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


# Q1-Q2-Critic Neural Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class ReplayBuffer(object):
    def __init__(self, gtr, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.gtr = gtr

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = self.gtr.integers(low=0, high=len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer
            state, next_state, action, reward, done = replay_buffer.sample(batch_size)
            original_action = action.copy()
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(original_action).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def save(agent, filename, directory):
    torch.save(agent.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(agent.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    torch.save(agent.actor_target.state_dict(), '%s/%s_actor_t.pth' % (directory, filename))
    torch.save(agent.critic_target.state_dict(), '%s/%s_critic_t.pth' % (directory, filename))


def td3_train(agent, env, rng, action_dim, n_episodes=3600, save_every=10):
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    time_start = time.time()  # Init start time
    replay_buf = ReplayBuffer(rng)  # Init ReplayBuffer

    timestep_after_last_save = 0
    total_timesteps = 0

    low = env.action_space.low
    high = env.action_space.high

    print('Low in action space: ', low, ', High: ', high, ', Action_dim: ', action_dim)

    for i_episode in range(1, n_episodes + 1):

        timestep = 0
        total_reward = 0

        # Reset environment
        state = env.reset()
        done = False

        while True:

            # Select action randomly or according to policy
            if total_timesteps < start_timestep:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state))
                if std_noise != 0:
                    shift_action = np.random.normal(0, std_noise, size=action_dim)
                    action = (action + shift_action).clip(low, high)

            # Perform action
            new_state, reward, done, _ = env.step(action)
            done_bool = 0 if timestep + 1 == env._max_episode_steps else float(done)
            total_reward += reward  # full episode reward

            # Store every timestep in replay buffer
            replay_buf.add((state, new_state, action, reward, done_bool))
            state = new_state

            timestep += 1
            total_timesteps += 1
            timestep_after_last_save += 1

            if done:  # done ?
                break  # save score

        scores_deque.append(total_reward)
        scores_array.append(total_reward)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        s = int(time.time() - time_start)
        print('Ep. {}, Timestep {},  Ep.Timesteps {}, Score: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02} ' \
              .format(i_episode, total_timesteps, timestep,
                      total_reward, avg_score, s // 3600, s % 3600 // 60, s % 60))

        agent.train(replay_buf, timestep)

        # Save episode if more than save_every=5000 timesteps
        if timestep_after_last_save >= save_every:
            timestep_after_last_save %= save_every
            save(agent, 'checkpnt_seed_88', 'Models')

        if len(scores_deque) == 100 and np.mean(scores_deque) >= 300.5:
            print('Environment solved with Average Score: ', np.mean(scores_deque))
            break

    return scores_array, avg_scores_array


def play(env, agent, n_episodes):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0

        time_start = time.time()

        while True:
            action = agent.select_action(np.array(state))
            env.render()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break

        s = int(time.time() - time_start)

        scores_deque.append(score)
        scores.append(score)

        print('Episode {}\tAverage Score: {:.2f},\tScore: {:.2f} \tTime: {:02}:{:02}:{:02}' \
              .format(i_episode, np.mean(scores_deque), score, s // 3600, s % 3600 // 60, s % 60))


def main(task_name):
    env = gym.make(task_name)#'BipedalWalkerHardcore-v3'/BipedalWalker-v3
    # Set seeds
    seed = 2022
    env.action_space.np_random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = TD3(state_dim, action_dim, max_action)

    scores, avg_scores = td3_train(agent=agent, env=env, rng=rng, action_dim=action_dim)
    save(agent, 'chpnt_2022_seed', 'hard_bipedal')
    print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores, label="Score")
    plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg on 100 episodes")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.ylabel('Score')
    plt.xlabel('Episodes #')
    plt.show()
