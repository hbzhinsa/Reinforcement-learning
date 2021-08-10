#%%
#https://www.marketwatch.com/investing/stock/ipgp/download-data?
#pip install gym-anytrading
import gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines import A2C
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#%%
df=pd.read_csv('data/gmedata.csv')
df['Data']=pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)
df.head()
# %%
env=gym.make('stocks-v0',df=df,frame_bound=(10,100),window_size=5)
env.action_space
#%%
state=env.reset()
while True:
    action=env.action_space.sample()
    n_state,reward,done,info=env.step(action)
    if done:
        print('info', info)
        break
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()
# %%
# %%
