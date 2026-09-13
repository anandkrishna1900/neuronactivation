"""Detailed diagnostic: trace bird trajectory with hand-designed controller."""
import sys
sys.path.insert(0, "src")
from flymind.environment.flappy import FlappyEnvironment
from flymind.agent.flappy_agent import HandDesignedFlapAgent

env = FlappyEnvironment(seed=42)
agent = HandDesignedFlapAgent(offset=40.0)
state = env.reset(seed=42)
agent.reset()
done = False
step = 0
while not done and step < 200:
    action = agent.act(state)
    nearest = None
    for p in state.pipes:
        if p["x"] > 80.0:
            if nearest is None or p["x"] < nearest["x"]:
                nearest = p
    gc = nearest["gap_center"] if nearest else None
    nx = nearest["x"] if nearest else None
    if step % 10 == 0 or done or (nearest and abs(state.bird_y - gc) < 20):
        print(f"step={step:3d} bird_y={state.bird_y:6.1f} vy={state.bird_vy:5.2f} action={action} pipe_x={nx:.0f} gap={gc:.0f} diff={state.bird_y - gc if gc else 0:.0f}")
    state, reward, done, info = env.step(action)
    step += 1
print(f"Final: score={info['score']} steps={info['step']}")
