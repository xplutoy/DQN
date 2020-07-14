import torch.optim as optim
from tensorboardX import SummaryWriter

from atari_wrappers import get_env
from models import *

LR = 2e-5
GAMMA = 0.99
N_FRAMES = 10 ** 8
BATCH_SIZE = 32
REPLAY_SIZE = 10000

MODEL_NAME = 'dqn_double'

SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

env_id = "PongNoFrameskip-v4"
env = get_env(env_id)
save_file_name = env_id + "-" + MODEL_NAME + ".pth"

replay_buffer = ReplayBuffer(REPLAY_SIZE)
net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(DEVICE)
tgt_net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(DEVICE)
trainer = optim.Adam(net.parameters(), lr=LR, betas=[0.5, 0.99])
writer = SummaryWriter(comment=MODEL_NAME)

episode_reward = 0
last_100_rewards = deque(maxlen=100)
best_mean_reward = None


def calc_td_loss(batch):
    state, action, reward, next_state, done = to_tensor(batch)
    q_value = net(state).gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value, _ = tgt_net.optimal_q_and_action(next_state)
    expected_q_value = reward + GAMMA * next_q_value.detach() * (1 - done.float())

    return nn.MSELoss()(q_value, expected_q_value)


state = env.reset()
for frame_idx in range(N_FRAMES):
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    if random.random() > epsilon:
        state_v = torch.tensor(state).unsqueeze(0).to(DEVICE)
        _, action = net.optimal_q_and_action(state_v)
        action = action.detach().item()
    else:
        action = random.randrange(env.action_space.n)
    next_state, reward, is_done, _ = env.step(action)
    replay_buffer.append(state, action, reward, next_state, is_done)
    episode_reward += reward

    if is_done:
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
        writer.add_scalar('mean_reward', mean_reward, frame_idx)
        if best_mean_reward is None or best_mean_reward < mean_reward:
            best_mean_reward = mean_reward
            torch.save(net.state_dict(), save_file_name)
        print("frame_idx: %d loss: %.3f mean_reward: %.3f best_mean_reward: %.3f" % (
            frame_idx, loss.item(), mean_reward, best_mean_reward))

        if mean_reward > 20:  # 停时条件
            print('Solved!!')
            break

writer.close()
