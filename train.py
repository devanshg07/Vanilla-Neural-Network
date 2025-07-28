import neuron
import math, random, sys, csv
from matplotlib import pyplot
from tqdm import tqdm

NUM_CLASSES = 5
STEP_SIZE = 0.03
NUM_EPOCHS = 100

features_list = []
targets_list = []

def progress_bar(it, desc):
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
        features = [float(r[0]), float(r[1]), float(r[2]), float(r[4])]
        features_list.append(features)
        targets_list.append(float(r[3]))  # GPA is now the 4th column
print(f"Input size: {len(features_list[0])}")

dataset_size = len(targets_list)
split_index = math.floor(dataset_size * 0.8)
X_train = features_list[:split_index]
y_train = targets_list[:split_index]
X_val = features_list[split_index:]
y_val = targets_list[split_index:]

model = neuron.MLP((4, "tanh"), (32, "relu"), (16, "tanh"), (8, "relu"), (1, "relu"))

parameters = model.get_parameters()
print(f"Parameters: {len(parameters)}")

train_mae_history = []
val_mae_history = []
best_val_error = float('inf')

for i in range(NUM_EPOCHS):
    epoch_X, epoch_y = zip(*random.sample(list(zip(features_list, targets_list)), dataset_size))

    print(f"\nEpoch: {i}")
    train_predictions = [model(X_train[i]) for i in progress_bar(X_train, "Forward Pass")]
    loss_train = neuron.mean_squared_error(y_train, train_predictions)
    mae_train = neuron.mean_abs_error(y_train, train_predictions)

    print(f"\tTraining Error: {mae_train}")

    model.zero_grad()
    loss_train.backward()
    model.nudge(STEP_SIZE)

    val_predictions = [model(X_val[i]) for i in progress_bar(X_val, "Validating")]
    loss_val = neuron.mean_squared_error(y_train, val_predictions)
    mae_val = neuron.mean_abs_error(y_val, val_predictions)

    train_mae_history.append(mae_train)
    pyplot.plot(range(i+1), train_mae_history, color="red", label="Training Error")
    val_mae_history.append(mae_val)
    pyplot.plot(range(1, i+2), val_mae_history, color="blue", label="Validation Error")
    pyplot.legend(["Training Error", "Validation Error"])
    pyplot.pause(0.001)
    print(f"\tValidation Error: {mae_val}")

train_predictions = [model(x) for x in X_train]
loss_train = neuron.mean_squared_error(y_train, train_predictions)

# Make a prediction with the trained model
sample_input = [17, 26, 3, 2]  # Example: Age, StudyTimeWeekly, Absences, TotalActivities
gpa_prediction = model(sample_input)
print(f"Predicted GPA for input {sample_input}: {gpa_prediction.data:.2f}")
