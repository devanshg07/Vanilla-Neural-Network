import neuron
import math, random, sys, csv, dill
from matplotlib import pyplot
from tqdm import tqdm

NUM_CLASSES = 5
STEP_SIZE = 0.001
NUM_EPOCHS = 50

all_x = []
all_y = []

def progress_iter(it, desc):
    return tqdm(range(len(it)),
                desc=f'\t{desc}',
                unit=" batches",
                file=sys.stdout,
                colour="GREEN",
                bar_format="{desc}: {percentage:0.2f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}]")

with open('dataset/Student_performance_data.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    data = list(reader)
    print(f"Data size: {len(data)}")
    random.shuffle(data)
    for r in data:
        # New order: Age, StudyTimeWeekly, Absences, GPA, TotalActivities
        # Input features: Age, StudyTimeWeekly, Absences, TotalActivities
        features = [float(r[0]), float(r[1]), float(r[2]), float(r[4])]
        all_x.append(features)
        all_y.append(float(r[3]))  # GPA is now the 4th column
print(f"Input size: {len(all_x[0])}")

SAMPLE_SIZE = len(all_y)
TRAIN_SPLIT = math.floor(SAMPLE_SIZE * 0.8)
train_x = all_x[:TRAIN_SPLIT]
train_y = all_y[:TRAIN_SPLIT]
valid_x = all_x[TRAIN_SPLIT:]
valid_y = all_y[TRAIN_SPLIT:]

# Network architecture for 4 input features
nn = neuron.MLP((4, "tanh"), (32, "relu"), (16, "tanh"), (8, "relu"), (1, "relu"))

parameters = nn.get_parameters()
print(f"Parameters: {len(parameters)}")

train_losses = []
valid_losses = []
best_valid_error = float('inf')

for i in range(NUM_EPOCHS):
    epoch_x, epoch_y = zip(*random.sample(list(zip(all_x, all_y)), SAMPLE_SIZE))

    print(f"\nEpoch: {i}")
    train_pred_y = [nn(train_x[i]) for i in progress_iter(train_x, "Forward Pass")]
    train_loss = neuron.mean_squared_error(train_y, train_pred_y)
    train_abs_error = neuron.mean_abs_error(train_y, train_pred_y)

    print(f"\tTraining Error: {train_abs_error}")

    nn.zero_grad()
    train_loss.backward()
    nn.nudge(STEP_SIZE)

    valid_pred_y = [nn(valid_x[i]) for i in progress_iter(valid_x, "Validating")]
    valid_loss = neuron.mean_squared_error(train_y, valid_pred_y)
    valid_abs_error = neuron.mean_abs_error(valid_y, valid_pred_y)

    train_losses.append(train_abs_error)
    pyplot.plot(range(i+1), train_losses, color="red", label="Training Error")
    valid_losses.append(valid_abs_error)
    pyplot.plot(range(1, i+2), valid_losses, color="blue", label="Validation Error")
    pyplot.legend(["Training Error", "Validation Error"])
    pyplot.pause(0.001)
    print(f"\tValidation Error: {valid_abs_error}")
    # No model saving here

# After all epochs, save the final model
with open('best_model.pkl', 'wb') as f:
    dill.dump(nn, f)
print(f"Model saved after {NUM_EPOCHS} epochs.")

train_pred_y = [nn(x) for x in train_x]
train_loss = neuron.mean_squared_error(train_y, train_pred_y)