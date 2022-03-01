import numpy as np
import torch
from collections import deque

MAX_REPLAY_BUFFER = 100000


class ReplayBuffer:

	def __init__(self, device, rng):
		self.buffer = deque(maxlen=MAX_REPLAY_BUFFER)
		self.maxSize = MAX_REPLAY_BUFFER
		self.device = device
		self.len = 0
		self.rng = rng

	def sample(self, batch_size):
		"""
		samples a random batch from the replay memory buffer
		:param batch_size: batch size
		:return: batch (numpy array)
		"""
		batch_size = min(batch_size, self.len)
		indices = self.rng.integers(low=0, high=len(self.buffer), size=batch_size)
		cur_state = torch.tensor(np.float32([self.buffer[i][0] for i in indices])).to(self.device)
		action = torch.tensor(np.float32([self.buffer[i][1] for i in indices])).to(self.device)
		reward = torch.tensor(np.float32([self.buffer[i][2] for i in indices])).to(self.device)
		next_state = torch.tensor(np.float32([self.buffer[i][3] for i in indices])).to(self.device)
		done = torch.tensor(np.float32([self.buffer[i][4] for i in indices])).to(self.device)
		return cur_state, action, reward, next_state, done

	def add(self, cur_state, action, reward, next_state, done):
		"""
		adds a particular transaction in the memory buffer
		:param cur_state: current state
		:param action: action taken
		:param reward: reward received
		:param next_state: next state
		:param done: done
		:return:
		"""
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append((cur_state, action, reward, next_state, done))
