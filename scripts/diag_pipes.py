"""Debug pipe positions and sensor activations."""
import sys
sys.path.insert(0, "src")
import numpy as np
from flymind.environment.flappy import FlappyEnvironment
from flymind.environment.flappy_sensor import FlappyVisualSensor

env = FlappyEnvironment(seed=42)
state = env.reset(seed=42)
sensor = FlappyVisualSensor()

print("Pipes at step 0:")
for p in state.pipes:
    dx = p["x"] - 80.0
    print(f"  x={p['x']:.0f} gap={p['gap_center']:.0f} dx={dx:.0f}")

acts = sensor.sense(state)
print(f"Sensor: {acts.round(3)}")
print(f"Total: {acts.sum():.3f}")
