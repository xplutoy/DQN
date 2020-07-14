from collections import namedtuple

import torch.optim as optim

from atari_wrappers import get_env
from models import *

K = 10
P = 0.5
LR = 3e-4
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 100000
N_FRAMES = 50000000

SYNC_TARGET_FRAMES = 1000

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 1000000

Experience = namedtuple('Experience', 'state, action, reward, next_state, mask, done')

env_id = "PongNoFrameskip-v4"
model_name = "BooststrappedDQN"
save_file_name = env_id + "-" + model_name + ".pth"

env = get_env(env_id)
replay_buffer = ExperimentReplayBuffer(REPLAY_SIZE)
net = BootstrappedDQN(env.observation_space.shape, env.action_space.n, K)
tgt_net = BootstrappedDQN(env.observation_space.shape, env.action_space.n, K)
# trainer = optim.Adam(net.parameters(), lr=LR, betas=[0.5, 0.99])
trainer = optim.RMSprop(net.parameters(), lr=LR, momentum=0.95)

episode_reward = 0
last_100_rewards = deque(maxlen=100)


def calc_td_loss(experiences):
    """
    double dqn的跟新方式
    :param experiences: list of experience
    :return:
    """
    state, action, reward, next_state, mask, done = to_tensor(zip(*experiences))
    action = action.view(1, -1, 1).repeat(K, 1, 1)
    q_value = net(state).gather(-1, action).squeeze(-1)
    alt_action = net.optimal_q_and_action(next_state)[1].unsqueeze(-1)
    next_q_value = tgt_net(next_state).gather(-1, alt_action).squeeze(-1)
    expected_q_value = reward.unsqueeze(0) + GAMMA * next_q_value.detach() * (1 - done.unsqueeze(0).float())
    mask = mask.t().float()
    return (mask * (q_value - expected_q_value) ** 2).mean()


k = random.randint(0, K - 1)
state = env.reset()
for frame_idx in range(N_FRAMES):
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    if random.random() > epsilon:
        state_v = torch.tensor(state).unsqueeze(0).to(DEVICE)
        _, action = net.optimal_q_and_action(state_v)
        action = action[k].detach().item()
    else:
        action = random.randrange(env.action_space.n)

    next_state, reward, is_done, _ = env.step(action)
    mask = np.random.binomial(1, P, K)
    replay_buffer.append(Experience(state, action, reward, next_state, mask, is_done))
    episode_reward += reward

    if is_done:
        k = random.randint(0, K - 1)
        state = env.reset()
        last_100_rewards.append(episode_reward)
        episode_reward = 0
    else:
        state = next_state

    if len(replay_buffer) < BATCH_SIZE or len(last_100_rewards) == 0:
        continue

    # update
    loss = calc_td_loss(replay_buffer.sample(BATCH_SIZE))
    trainer.zero_grad()
    loss.backward()
    trainer.step()

    if (frame_idx + 1) % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())
        mean_reward = np.mean(last_100_rewards)
        print("frame_idx: %d loss: %.3f mean_reward: %.3f" % (frame_idx, loss.item(), mean_reward))

        if mean_reward > 20:  # 停时条件
            torch.save(net.state_dict(), save_file_name)
            print('Solved!!')
            break
