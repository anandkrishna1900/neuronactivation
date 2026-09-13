"""Quick diagnostic: trace bird trajectory through the Flappy environment."""
import sys
sys.path.insert(0, "src")
from flymind.environment.flappy import FlappyEnvironment
from flymind.agent.flappy_agent import HandDesignedFlapAgent

env = FlappyEnvironment(seed=42)
agent = HandDesignedFlapAgent(offset=10.0)
state = env.reset(seed=42)
agent.reset()
done = False
step = 0
while not done and step < 200:
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    if step % 20 == 0 or done:
        print(f"step={step:3d} bird_y={state.bird_y:6.1f} vy={state.bird_vy:5.2f} action={action} pipes={len(state.pipes)} score={state.score}")
    step += 1
print(f"Final: score={info['score']} steps={info['step']} alive={state.alive}")
