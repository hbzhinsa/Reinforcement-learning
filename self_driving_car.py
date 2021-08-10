# brew install swig
# pip install 'gym[box2d]' pyglet
#%%
import os
import gym # openai
from stable_baselines3 import PPO # 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#%%
environment_name='CarRacing-v0'
env=gym.make(environment_name)
# %%
episodes =2
for episode in range(1, episodes+1):
    state=env.reset()
    done=False
    score=0
    while not done:
        env.render()
        action=env.action_space.sample()
        n_state,reward,done,info=env.step(action)
        score+=reward
    print('Episode:{}, Score:{}'.format(episode,score))
env.close()
# %%
env=DummyVecEnv([lambda: env])
log_path=os.path.join('Training','Logs')
model=PPO('CnnPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=100)

# %%
ppo_path=os.path.join('Training','SavedModels','PPO_2m_Driving_model')
model.save(ppo_path)
# %%
model=PPO.load(ppo_path,env=env)
# %%
evaluate_policy(model,env,n_eval_episodes=1,render=True)

# %%
