
# cart_pole
#https://stable-baselines3.readthedocs.io/en/master/
#https://gym.openai.com/docs/
#%%
import os
import gym # openai
from stable_baselines3 import PPO # 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import subprocess
# test cart_pole game
#%%
environment_name='CartPole-v0'
env=gym.make(environment_name)
env
#%%
episodes =5
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
log_path=os.path.join('Training','Logs')
log_path
# %%
env=gym.make(environment_name)
env=DummyVecEnv([lambda: env])
model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
# %%
model.learn(total_timesteps=20000)
# %%
PPO_path=os.path.join('Training','SavedModels','PPO_Model_cartpole')
model.save(PPO_path)
del model
#%%
model=PPO.load(PPO_path,env=env)
# %%
# Test model
episodes =1
for episode in range(1, episodes+1):
    obs=env.reset()
    done=False
    score=0
    while not done:
        env.render()
        action,_=model.predict(obs)
        obs,reward,done,info=env.step(action)
        score+=reward
    print('Episode:{}, Score:{}'.format(episode,score))
env.close()

# %%
training_log_path=os.path.join(log_path,'PPO_1')
# os.system('tensorborad --logdir={training_log_path}')
# %%
