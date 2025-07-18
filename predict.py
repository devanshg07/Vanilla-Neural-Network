import anndy

# Define the model architecture (must match training)
nn = anndy.MLP((8, "tanh"), (16, "relu"), (8, "tanh"), (2, "relu"), (1, "relu"))

# Example input (replace with your actual values)
sample_input = [540.0,0.0,0.0,162.0,2.5,1055.0,676.0,28,61.887365759999994]  # Must be 8 features

# Make a prediction
prediction = nn(sample_input)

print("Prediction:", prediction) 