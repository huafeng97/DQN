from keras import layers, Model, Input
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt


class DQN:
    def __init__(self, n_features, n_actions):
        self.n_features = n_features
        self.n_actions = n_actions
        # 算法超参数设置
        self.lr = 0.01
        self.gamma = 0.9
        self.replace_target_iter = 300
        self.memory_size = 5000
        self.batch_size = 60
        self.epsilon_max = 0.99
        self.epsilon_increment = 0.05  # 不断减小随机选择的概率
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        # 记忆[s,a,r,s_]：根据提议s长度为2，a长度为1，r长度为1，故总长度为2*2+1+1
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        # 初始化两个网络模型
        self.model_eval = self.create_eval(n_actions, n_features)
        self.model_target = self.create_target(n_actions, n_features)
        self.model_eval.compile(optimizer=RMSprop(lr=self.lr), loss='mse')
        # 记录mse(TD_error)
        self.cost_his = []

    def create_eval(self, n_actions, n_features):
        input_tensor = Input(shape=(n_features,))
        x = layers.Dense(32, activation='relu')(input_tensor)
        x = layers.Dense(32, activation='relu')(x)
        output_tensor = layers.Dense(n_actions)(x)
        model = Model(input_tensor, output_tensor)
        model.summary()
        return model

    def create_target(self, n_actions, n_features):
        input_tensor = Input(shape=(n_features,))
        x = layers.Dense(32, activation='relu', trainable=False)(input_tensor)
        x = layers.Dense(32, activation='relu', trainable=False)(x)
        output_tensor = layers.Dense(n_actions)(x)
        model = Model(input_tensor, output_tensor)
        model.summary()
        return model

    def store_transition(self, s, a, r, s_):
        # 存储第一条初始化
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))  # 水平叠加,例如:np.hstack(([1,2],[3,4],[5,6])) => array(1 2 3 4 5 6)
        index = self.memory_counter % self.memory_size  # 后续经历覆盖老经历
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.model_eval.predict(observation)
            action = np.argmax(action_value)  # 0，1，2，3表示动作
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        for eval_layer, target_layer in zip(self.model_eval.layers, self.model_target.layers):
            target_layer.set_weights(eval_layer.get_weights())

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]  # shuffle
        q_next = self.model_target.predict(batch_memory[:, -self.n_features:])
        q_eval = self.model_eval.predict(batch_memory[:, :self.n_features])

        q_target = q_eval.copy()  # array不需要deepcopy

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        # 为了方便矩阵相减，q_target - q_eval， 它们的最大q值要放在同一个矩阵位置
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.learn_step_counter % self.replace_target_iter == 0:
            # 每隔若干步学习一次
            self._replace_target_params()

        cost = self.model_eval.train_on_batch(batch_memory[:, :self.n_features], q_target)
        self.cost_his.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('mse')
        plt.xlabel('training steps')
        plt.show()

