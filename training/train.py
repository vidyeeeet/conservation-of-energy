import numpy as np
import torch
import torch.nn as nn
import json
import os
import argparse
from datetime import datetime

def simulate_spring(k, m, dt, n_trials, n_steps):
    """Simulate spring-mass system with diverse initial conditions."""
    positions = []
    velocities = []

    for _ in range(n_trials):
        x = np.random.uniform(-1, 1)
        v = np.random.uniform(-1, 1)

        for _ in range(n_steps):
            a = -(k / m) * x
            x = x + v * dt + 0.5 * a * dt**2
            a_new = -(k / m) * x
            v = v + 0.5 * (a + a_new) * dt
            a = a_new

            positions.append(x)
            velocities.append(v)

    return positions, velocities


def build_dataset(positions, velocities):
    """Build input/target pairs from trajectory data."""
    inputs = []
    targets = []

    for i in range(len(positions) - 1):
        inputs.append([positions[i], velocities[i]])
        targets.append(positions[i + 1])

    inputs = torch.tensor(inputs).float()
    targets = torch.tensor(targets).float().unsqueeze(1)

    return inputs, targets


def build_model():
    """Build the world model architecture."""
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )


def train(model, inputs, targets, lr, n_steps):
    """Train the model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    losses = []

    for i in range(n_steps):
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 500 == 0:
            print(f"Step {i:5d} | Loss: {loss.item():.6f}")

    return losses


def save_model(model, losses, config):
    """Save model weights and metadata as JSON."""
    os.makedirs('models', exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy().tolist()

    model_data = {
        'timestamp': timestamp,
        'architecture': '2-64-1',
        'activation': 'tanh',
        'final_loss': losses[-1],
        'config': config,
        'weights': weights
    }

    filename = f'models/model_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(model_data, f)

    print(f"\nSaved: {filename}")
    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train spring world model')
    parser.add_argument('--k', type=float, default=1.0, help='Spring stiffness')
    parser.add_argument('--m', type=float, default=1.0, help='Mass')
    parser.add_argument('--dt', type=float, default=0.05, help='Timestep')
    parser.add_argument('--trials', type=int, default=200, help='Number of trajectories')
    parser.add_argument('--steps', type=int, default=200, help='Steps per trajectory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Training steps')
    args = parser.parse_args()

    config = {
        'k': args.k,
        'm': args.m,
        'dt': args.dt,
        'n_trials': args.trials,
        'n_steps': args.steps,
        'lr': args.lr,
        'epochs': args.epochs
    }

    print("Simulating spring trajectories...")
    positions, velocities = simulate_spring(
        args.k, args.m, args.dt, args.trials, args.steps
    )

    print(f"Dataset size: {len(positions)} timesteps")
    inputs, targets = build_dataset(positions, velocities)

    print("Building model...")
    model = build_model()

    print("Training...")
    losses = train(model, inputs, targets, args.lr, args.epochs)

    print(f"\nFinal loss: {losses[-1]:.6f}")
    save_model(model, losses, config)