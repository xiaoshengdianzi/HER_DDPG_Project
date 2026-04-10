import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，防止Qt报错
import matplotlib.pyplot as plt
from grid_world_env import GridWorldEnv
from her_replay import ReplayBuffer_Trajectory, Trajectory
from ddpg import DDPG

# 参数配置
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    actor_lr = 1e-3
    critic_lr = 1e-3
    hidden_dim = 128
    state_dim = 4
    action_dim = 2
    action_bound = 1
    sigma = 0.1
    tau = 0.005
    gamma = 0.98
    num_episodes = 2000
    n_train = 20
    batch_size = 256
    minimal_episodes = 200
    buffer_size = 10000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(0)

    env = GridWorldEnv()
    replay_buffer = ReplayBuffer_Trajectory(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr,
                 sigma, tau, gamma, device)
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(action, next_state, reward, done)
                    state = next_state
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= minimal_episodes:
                    for _ in range(n_train):
                        transition_dict = replay_buffer.sample(batch_size, True)
                        agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1),
                                     'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    # 计算滑动平均奖励
    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(8, 5))
    plt.plot(episodes_list, return_list, label='Episode Return', color='blue', alpha=0.4)
    ma_returns = moving_average(return_list)
    plt.plot(episodes_list[:len(ma_returns)], ma_returns, label='Moving Average (window=50)', color='red', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG with HER on GridWorld')
    plt.legend()
    plt.savefig('images/train_result.png')  # 保存图片，不弹窗
    # 训练结束后效果测试
    print("\n========== 测试训练效果 ==========")
    evaluate_policy(agent, env, episodes=100)

# 效果测试函数
def evaluate_policy(agent, env, episodes=100):
    success_count = 0
    total_return = 0
    all_trajs = []  # 存储所有轨迹
    all_goals = []  # 存储所有目标
    for _ in range(episodes):
        state = env.reset()
        traj_x, traj_y = [state[0]], [state[1]]
        episode_return = 0
        done = False
        while not done:
            state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(agent.device)
            action = agent.actor(state_tensor).detach().cpu().numpy()[0]
            next_state, reward, done = env.step(action)
            traj_x.append(next_state[0])
            traj_y.append(next_state[1])
            episode_return += reward
            state = next_state
        all_trajs.append((traj_x, traj_y))
        all_goals.append(env.goal.copy())
        total_return += episode_return
        if reward == 0:  # 到达目标
            success_count += 1
    print(f"平均回报: {total_return / episodes:.2f}")
    print(f"成功率: {success_count / episodes * 100:.1f}%")
    # 绘制多条轨迹
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    for i, (traj_x, traj_y) in enumerate(all_trajs):
        plt.plot(traj_x, traj_y, marker='o', alpha=0.3, label='Trajectory' if i==0 else None)
    for i, goal in enumerate(all_goals):
        plt.scatter(goal[0], goal[1], c='red', marker='*', s=100, alpha=0.5, label='Goal' if i==0 else None)
    plt.xlim(0, env.size)
    plt.ylim(0, env.size)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agent Trajectories to Goals')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/trajectory.png')
    plt.close()

if __name__ == '__main__':
    main()
