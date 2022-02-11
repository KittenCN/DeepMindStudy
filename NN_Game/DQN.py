import random
import gym
import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Sequential):
    def __init__(self):
        super(QNet, self).__init__(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

class Game:
    def __init__(self, exp_pool_size, explore):
        self.env = gym.make('CartPole-v1')
        self.exp_pool = []
        self.exp_pool_size = exp_pool_size
        self.q_net = QNet().to(device)
        self.explore = explore
        self.loss_fn = nn.MSELoss()
        self.opt = optim.Adam(self.q_net.parameters())

    def __call__(self):
        is_render = False
        avg = 0
        while True:
            # 数据采样
            state = self.env.reset()
            R = 0
            while True:
                if is_render:
                    self.env.render()
                if len(self.exp_pool) >= self.exp_pool_size:
                    self.exp_pool.pop(0)
                    self.explore += 1e-7
                    if torch.rand(1) > self.explore:
                        action = self.env.action_space.sample()
                    else:
                        _state = torch.tensor(state, dtype=torch.float32).to(device)
                        Qs = self.q_net(_state[None, ...]).to(device)
                        action = torch.argmax(Qs, 1)[0].item()
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)
                R += reward
                self.exp_pool.append([state, reward, action, next_state, done])
                state = next_state

                if done:
                    avg = 0.95 * avg + 0.05 * R
                    print(avg, R)
                    if avg > 1000:
                        is_render = True
                    break
            # 训练
            if len(self.exp_pool) >= self.exp_pool_size:
                exps = random.choices(self.exp_pool, k=100)
                _state = torch.tensor([exp[0] for exp in exps]).float().to(device)
                _reward = torch.tensor([[exp[1]] for exp in exps]).to(device)
                _action = torch.tensor([[exp[2]] for exp in exps]).to(device)
                _next_state = torch.tensor([exp[3] for exp in exps]).float().to(device)
                _done = torch.tensor([[int(exp[4])] for exp in exps]).to(device)

                # 预测值
                _Qs = self.q_net(_state).to(device)
                _Q = torch.gather(_Qs, 1, _action)
                # 目标值
                _next_Qs = self.q_net(_next_state).to(device)
                _max_Q = torch.max(_next_Qs, dim=1, keepdim=True)[0]
                _target_Q = _reward + (1 - _done) * 0.9 * _max_Q

                loss = self.loss_fn(_Q, _target_Q.detach())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if __name__ == '__main__':
    g = Game(10000, 0.9)
    g()