import sys
import numpy as np
import gymnasium as gym
import time
from stable_baselines3 import PPO
from zxy9_biomechanical_model import simulation as sim


if len(sys.argv) < 2 or (len(sys.argv) > 2 and sys.argv[1] != 'new'):
	print("Usage:\n" + \
		"\ttrain.py <model-name>\n" + \
		"\t\tContinue training of a model\n" + \
		"\ttrain.py new <model-name>\n" + \
		"\t\tStart training of a new model")
	exit()


# -- reward functions -- #


# standing task
top_of_torso_score_multiplier = 200
top_of_torso_ideal_height = 0.54 + 0.418 + 0.445

def rew_stand(simu, last_call):
	abs_angle_sum = np.sum(np.abs(simu.thetas))
	top_of_torso_height = simu.positions[1]
	top_of_torso_loss =  \
		top_of_torso_score_multiplier * \
		abs(top_of_torso_ideal_height - top_of_torso_height)
	h_loss = 10 * (simu.positions[2]) ** 2
	
	if simu.positions[0 * 2 + 1] <= 0 \
		or simu.positions[1 * 2 + 1] <= 0 \
		or simu.positions[2 * 2 + 1] <= 0 \
		or simu.positions[3 * 2 + 1] <= 0 \
		or simu.positions[4 * 2 + 1] <= 0 \
		or simu.positions[5 * 2 + 1] <= 0 \
		or simu.positions[6 * 2 + 1] <= 0 \
		or simu.positions[9 * 2 + 1] <= 0:
		score = - abs_angle_sum - 3 * top_of_torso_score_multiplier * \
			top_of_torso_ideal_height - h_loss
	else:
		score = - abs_angle_sum - top_of_torso_loss - h_loss
	
	return score


# jump task
max_tt_height = 0.0
max_h_horizontal = 0.0

def rew_jump(simu, last_call):
	global max_tt_height
	global max_h_horizontal
	max_tt_height = max(max_tt_height, simu.positions[1])
	max_h_horizontal = max(max_h_horizontal, abs(simu.positions[0]))
	if last_call:
		max_tt_height_copy = max_tt_height
		max_h_horizontal_copy = max_h_horizontal
		max_tt_height = 0.0
		max_h_horizontal = 0.0
		return 1000 * (max_tt_height_copy - \
			max_h_horizontal_copy)
	return 0.0


# running task
def rew_run(simu, last_call):
	if simu.positions[0 * 2 + 1] <= 0 \
		or simu.positions[1 * 2 + 1] <= 0 \
		or simu.positions[2 * 2 + 1] <= 0 \
		or simu.positions[3 * 2 + 1] <= 0 \
		or simu.positions[4 * 2 + 1] <= 0 \
		or simu.positions[5 * 2 + 1] <= 0 \
		or simu.positions[6 * 2 + 1] <= 0 \
		or simu.positions[9 * 2 + 1] <= 0:
		return 0.0
	return 50 * simu.positions[2]


# -- make simulation with our reward functions (+2 possible future ones) -- #


s = sim.Simulation( \
	# body and start position files
	"RL-inputs/human/body.json", \
	"RL-inputs/human/start-position.json", \
	# reward functions
	env_reward_functions=[rew_stand, rew_jump, rew_run], \
	env_stop_times=[3.0, 5.0, 6.0], env_potential_reward_num=5, \
	# randomize the strengths, positions, velocities of each episode
	env_strength_low=0.2, env_strength_high=1.0, env_thetas_diff=5.0, \
	env_velocities_diff=2.5 \
)


# -- load pre-existing or new model -- #


if sys.argv[1] == 'new':
	model_name = sys.argv[2]
	model = PPO("MultiInputPolicy", s, verbose=0, gamma=1.0)
else:
	model_name = sys.argv[1]
	model = PPO.load(sys.argv[1], verbose=0, env=s)


# -- train and periodically save to file -- #


while True:
	model.learn(total_timesteps=5_000_000, progress_bar=True)
	model.save(model_name)
	print("saved")
