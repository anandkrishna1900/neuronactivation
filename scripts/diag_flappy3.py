"""Diagnostic: trace bird trajectory with velocity-aware hand-designed controller."""
import sys
sys.path.insert(0, "src")
from flymind.environment.flappy import FlappyEnvironment
from flymind.agent.flappy_agent import HandDesignedFlapAgent

env = FlappyEnvironment(seed=42)
agent = HandDesignedFlapAgent(gap_offset=30.0)
state = env.reset(seed=42)
agent.reset()
done = False
step = 0
while not done and step < 300:
    action = agent.act(state)
    nearest = None
    for p in state.pipes:
        if p["x"] > 80.0:
            if nearest is None or p["x"] < nearest["x"]:
                nearest = p
    gc = nearest["gap_center"] if nearest else None
    if step % 30 == 0 or done or (gc and state.bird_y < gc - 30 and state.bird_vy < 0):
        print(f"step={step:3d} bird_y={state.bird_y:6.1f} vy={state.bird_vy:5.2f} action={action} gap={gc:.0f} diff={state.bird_y - gc if gc else 0:.0f}")
    state, reward, done, info = env.step(action)
    step += 1
print(f"Final: score={info['score']} steps={info['step']}")
