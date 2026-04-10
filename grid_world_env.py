import numpy as np
import random
import matplotlib.pyplot as plt

class GridWorldEnv:
    def __init__(self, size=5, goal_range=((3.5, 4.5), (3.5, 4.5)), distance_threshold=0.15, max_steps=50, action_bound=1):
        self.size = size
        self.goal_range = goal_range
        self.distance_threshold = distance_threshold
        self.max_steps = max_steps
        self.action_bound = action_bound
        self.state = None
        self.goal = None
        self.count = 0

    def reset(self):
        self.goal = np.array([
            random.uniform(self.goal_range[0][0], self.goal_range[0][1]),
            random.uniform(self.goal_range[1][0], self.goal_range[1][1])
        ])
        self.state = np.array([0.0, 0.0])
        self.count = 0
        return np.hstack((self.state, self.goal))

    def step(self, action):
        action = np.clip(action, -self.action_bound, self.action_bound)
        x = np.clip(self.state[0] + action[0], 0, self.size)
        y = np.clip(self.state[1] + action[1], 0, self.size)
        self.state = np.array([x, y])
        self.count += 1
        dis = np.linalg.norm(self.state - self.goal)
        reward = 0.0 if dis <= self.distance_threshold else -1.0
        done = dis <= self.distance_threshold or self.count >= self.max_steps
        return np.hstack((self.state, self.goal)), reward, done

    def render(self, mode='human'):
        plt.figure(figsize=(5, 5))
        plt.xlim(0, self.size)
        plt.ylim(0, self.size)
        # 绘制目标区域
        plt.gca().add_patch(plt.Rectangle((self.goal_range[0][0], self.goal_range[1][0]),
                                          self.goal_range[0][1] - self.goal_range[0][0],
                                          self.goal_range[1][1] - self.goal_range[1][0],
                                          fill=False, edgecolor='green', linestyle='--', linewidth=2, label='Goal Area'))
        # 绘制目标点
        plt.scatter(self.goal[0], self.goal[1], c='red', marker='*', s=200, label='Goal')
        # 绘制智能体当前位置
        plt.scatter(self.state[0], self.state[1], c='blue', marker='o', s=100, label='Agent')
        plt.legend()
        plt.title('GridWorld Environment')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
