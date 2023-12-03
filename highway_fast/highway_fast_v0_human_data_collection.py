import gymnasium as gym
import highway_env
from pynput import keyboard
import json
from datetime import datetime

now = datetime.now()


file = open('./data_highway_fast_v0/training_data_{}.json'.format(now.strftime("%m_%d_%Y_%H_%M_%S")),'w')




class KeyObj:
  keypressed = 1


def determine_pressed_arrow(key, keyobj):
  if key  == None:
    keyobj.keypressed = 1
  if key  == keyboard.Key.right:
    keyobj.keypressed = 3
  if key  == keyboard.Key.left:
    keyobj.keypressed = 4
  if key  == keyboard.Key.up:
    keyobj.keypressed = 0
  if key  == keyboard.Key.down:
    keyobj.keypressed = 2
  if key == keyboard.Key.esc:
    return False

def on_release(key, keyobj):
  keyobj.keypressed = 1

env = gym.make("highway-fast-v0", render_mode='rgb_array')

env.configure({
  "observation":{"type":"OccupancyGrid",
                 "features": ["presence", "vx", "vy",]},
  "action":{"type":"DiscreteMetaAction"},
  "simulation_frequency": 20
})

listener = keyboard.Listener(on_press = lambda event: determine_pressed_arrow(event, KeyObj), on_release= lambda event: on_release(event, KeyObj))

listener.start()

out_json = {}
i = 0
while listener.running:
  done = truncated = False
  obs, info = env.reset()
  i += 1
  simdata = []
  while not (done or truncated):
    obs_dict = {}
    obs_dict['input'] = KeyObj.keypressed
    
    obs, reward, done, truncated, info = env.step(KeyObj.keypressed)
    print(obs)
    obs_dict['obs'] = obs.tolist()
    obs_dict['reward'] = float(reward)
    simdata.append(obs_dict)
    env.render()
  out_json[i] = simdata
  
json_object = json.dumps(out_json)

file.write(json_object)

file.close()