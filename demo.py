import anndy
import math, random, sys, csv
from matplotlib import pyplot
from tqdm import tqdm
import grapher

NUM_CLASSES = 5
STEP_SIZE = 0.01  # Reduced learning rate for the different data characteristics
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


with open('dataset/cleaned_used_car_price_dataset.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    data = list(reader)
    print(f"Data size: {len(data)}")
    random.shuffle(data)
    for r in data:
        all_x.append([float(i) for i in r[:-1]])  # All features except the last one (price)
        all_y.append(float(r[-1]))  # Last column is the target (price)
print(f"Input size: {len(all_x[0])}")

SAMPLE_SIZE = len(all_y)
TRAIN_SPLIT = math.floor(SAMPLE_SIZE * 0.8)
train_x = all_x[:TRAIN_SPLIT]
train_y = all_y[:TRAIN_SPLIT]
valid_x = all_x[TRAIN_SPLIT:]
valid_y = all_y[TRAIN_SPLIT:]

# Updated network architecture for 19 input features
nn = anndy.MLP((19, "tanh"), (32, "relu"), (16, "tanh"), (8, "relu"), (1, "relu"))

parameters = nn.get_parameters()
print(f"Parameters: {len(parameters)}")

train_losses = []
valid_losses = []

for i in range(NUM_EPOCHS):

    # Create random mini-batch
    epoch_x, epoch_y = zip(*random.sample(list(zip(all_x, all_y)), SAMPLE_SIZE))


    print(f"\nEpoch: {i}")
    train_pred_y = [nn(train_x[i]) for i in progress_iter(train_x, "Forward Pass")]  # Forward pass
    train_loss = anndy.mean_squared_error(train_y, train_pred_y)
    train_abs_error = anndy.mean_abs_error(train_y, train_pred_y)

    print(f"\tTraining Error: {train_abs_error}")

    nn.zero_grad()
    train_loss.backward()  # Backward pass
    nn.nudge(STEP_SIZE)

    # VALIDATION

    valid_pred_y = [nn(valid_x[i]) for i in progress_iter(valid_x, "Validating")]

    valid_loss = anndy.mean_squared_error(train_y, valid_pred_y)
    valid_abs_error = anndy.mean_abs_error(valid_y, valid_pred_y)

    train_losses.append(train_abs_error)
    train_loss_line = pyplot.plot(range(i+1), train_losses, color="red", label="Training Error")
    valid_losses.append(valid_abs_error)
    valid_loss_line = pyplot.plot(range(1, i+2), valid_losses, color="blue", label="Validation Error")
    pyplot.legend(["Training Error", "Validation Error"])

    pyplot.pause(0.001)
    print(f"\tValidation Error: {valid_abs_error}")


train_pred_y = [nn(x) for x in train_x]
train_loss = anndy.mean_squared_error(train_y, train_pred_y)