import gymnasium as gym
import highway_env
from pynput import keyboard


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

def on_release(key, keyobj):
  keyobj.keypressed = 1

env = gym.make("highway-fast-v0", render_mode='rgb_array')

env.configure({
  "action":{"type":"DiscreteMetaAction"},
  "simulation_frequency": 20
})

listener = keyboard.Listener(on_press = lambda event: determine_pressed_arrow(event, KeyObj), on_release= lambda event: on_release(event, KeyObj))

listener.start()

while True:
  done = truncated = False
  obs, info = env.reset()
  print('new sim')
  while not (done or truncated):
    print(obs)
    obs, reward, done, truncated, info = env.step(KeyObj.keypressed)

    env.render()


