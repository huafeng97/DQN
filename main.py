from maze_env import Maze
from RL_brain import DQN


def run():
    step = 0
    for eposide in range(1000):
        # 初始化状态
        observation = env.reset()
        # 从当前状态出发，走一遍直到终结，记录过程中的经历
        while True:
            # 刷新环境
            env.render()
            # 根据当前状态选择行为
            action = RL.choose_action(observation)
            # 环境根据行为跳转状态和给出奖励
            observation_, reward, done = env.step(action)
            # 记忆库存储（s,a,r,s_)
            RL.store_transition(observation,action,reward,observation_)

            if (step > 200) and (step % 5 == 0):
                # 记忆库大于200时才开始学习，每5步学习一次
                RL.learn()

            observation = observation_
            if done: break
            step += 1
    print('game-over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = DQN(env.n_features, env.n_actions)
    env.after(100, run)
    env.mainloop()
    RL.plot_cost()
