#Variables
#Variables
k = 1.0
m = 1.0
dt = 0.05

clean_trajectory = []
velocity_trajectory = []

for trial in range(50):
    x = np.random.uniform(-1, 1)
    v = np.random.uniform(-1, 1)
    for i in range(200):
        a = -(k/m)*x
        x = x + v*dt + 0.5*a*dt**2
        a_new = -(k/m)*x
        v = v + 0.5*(a + a_new)*dt
        a = a_new
        clean_trajectory.append(x)
        velocity_trajectory.append(v)

inputs = []
targets = []

for i in range(len(clean_trajectory) - 1):
    inputs.append([clean_trajectory[i], velocity_trajectory[i]])
    targets.append(clean_trajectory[i+1])

inputs = torch.tensor(inputs).float()
targets = torch.tensor(targets).float().unsqueeze(1)

model = nn.Sequential(
    nn.Linear(2, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
losses = []

for i in range(5000):
    outputs = model(inputs)
    loss = loss_func(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if i % 100 == 0:
        print(i, loss.item())

plt.plot(losses)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Training Loss')
plt.show()