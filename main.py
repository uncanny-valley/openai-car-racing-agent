import gym
from gym.envs.box2d import CarRacing


env = CarRacing(
    grayscale=1,
    show_info_panel=0,
    discretize_actions='hard',
    frames_per_state=4,
    num_lanes=1,
    num_tracks=1
)

for _ in range(20):
    observation = env.reset()

    for t in range(100):
        env.render()
        print('observation:', observation)
        action = env.action_space.sample()
        print('action:', action)
        observation, reward, done, info = env.step(action)
        if done:
            print(f'Episode finished after {t+1} timesteps')
            break
    
env.close()