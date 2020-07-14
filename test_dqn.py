from atari_wrappers import get_env
from models import *
from utils import *

model_name = 'dqn_prioritization'
env_id = "SpaceInvadersNoFrameskip-v4"
env = get_env(env_id)
save_file_name = env_id + "-" + model_name + ".pth"

net = DQN(env.observation_space.shape, env.action_space.n).to(DEVICE)
net.load_state_dict(torch.load(save_file_name))

for _ in range(10):
    state = env.reset()
    env.render()
    while True:
        state_v = torch.tensor(state).unsqueeze(0).to(DEVICE)
        _, action = net.optimal_q_and_action(state_v)
        next_state, reward, is_done, _ = env.step(action.item())
        env.render()
        if is_done:
            break
        else:
            state = next_state
env.close()
