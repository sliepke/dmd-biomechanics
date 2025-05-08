import sys
import time

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from zxy9_biomechanical_model import simulation as sim

s = sim.Simulation( \
	"../../src/zxy9_biomechanical_model/inputs/human/body.json", \
	"../../src/zxy9_biomechanical_model/inputs/human/start-position.json", \
	rk4_timestep=1e-5, B_normal=1000 \
)
s.extension_c1s[6] *= 1.0
s.extension_c1s[7] *= 1.0


if len(sys.argv) < 2:
	print("enter the model to load as the first command line arg")
	exit()

model = PPO.load(sys.argv[1])

num_prints = 0

def act_callback(simu):
	global num_prints
	obs = {'state': np.concatenate((simu.positions, simu.velocities, \
		simu.flexion_c1s, simu.extension_c1s), dtype=np.float32), 'reward': 0}
	ret = model.predict(obs)[0]
	if num_prints > 4:
		return ret
	else:
		print(f"prediction: {ret}")
		num_prints += 1
		return ret


s.run(display=True, speed=0.1, activations_callback=act_callback, stop_time=3, width_px=1000, width_meters=4, point_radius_px=5, debug_at_stop=True)
time.sleep(1000)
