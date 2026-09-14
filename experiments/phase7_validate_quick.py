import sys
sys.path.insert(0, 'src')
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.agent.flappy_agent_phase7 import FlyMindRLAgent
from flymind.environment.flappy_reward import FlappyRewardShaper
from flymind.environment.curriculum import CurriculumManager
print('All imports OK')

graph = ConnectomeLoader.load_from_json('data/processed/cx_heading_v1.json')
net = NeuralNetwork(graph, synapse_scale=0.01)
agent = FlyMindRLAgent(network=net, seed=42)
print(f'Agent created. PEG: {len(agent.peg_indices)}, EPG: {len(agent.epg_indices)}, PFNd: {len(agent.pfnd_indices)}, PFNv: {len(agent.pfnv_indices)}')
print(f'W_motor: {agent.W_motor}')
print(f'Plastic synapses: {int(agent.plasticity_mask.sum())} / {int((net.synapses.raw_weights>0).sum())}')
agent.assert_biological_integrity()
print('Biological integrity assertions: PASSED')

from flymind.environment.flappy import FlappyEnvironment
env = FlappyEnvironment(seed=42, max_steps=50)
shaper = FlappyRewardShaper()
state = env.reset(seed=42)
agent.reset()
for i in range(10):
    action = agent.act(state)
    diag = agent.get_diagnostics()
    state, r, done, info = env.step(action)
    shaped_r = shaper.shape(r, info, done)
    agent.apply_step_reward(shaped_r)
    if done:
        break
ep_stats = agent.end_episode(0.5)
print(f'10-step test passed. flap_prob={diag["flap_prob"]:.3f}, peg_mean={diag["peg_mean"]:.4f}, pfnd_mean={diag["pfnd_mean"]:.4f}')
print(f'Motor grad norm: {ep_stats["motor_grad_norm"]:.6f}')
print('All tests PASSED')
