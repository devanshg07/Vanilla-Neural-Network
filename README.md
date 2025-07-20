# Pure Python Neural Network from Scratch

A neural network framework I built entirely from scratch using only Python and basic math—no PyTorch, NumPy, or TensorFlow. All computations, gradients, and training logic are hand-coded for educational clarity and transparency.

## 🎯 How It Works

### Architecture
- **Core**: Pure Python, no external ML or math libraries
- **Computation**: Custom classes for scalars, nodes, layers, and neural nets
- **Training**: Manual forward/backward propagation and gradient descent
- **Visualization**: Graphviz-based computation graph visualizer

### Process Flow
1. Load and preprocess tabular data (CSV)
2. Initialize a neural network with custom architecture
3. Train using hand-coded backpropagation and gradient updates
4. Make predictions with the trained model
5. (Optional) Visualize computation graph for any output

## 🛠️ Tools & Technologies Used

### Core
- **Python 3.x**
- **Standard Library**: math, random, csv, etc.
- **Graphviz**: For computation graph visualization
- **matplotlib**: For plotting training/validation error
- **tqdm**: For progress bars

### What I Did NOT Use
- No PyTorch
- No NumPy
- No TensorFlow
- No scikit-learn

## 🚀 How to Run

### Prerequisites
- Python 3.7+
- pip (for matplotlib, tqdm, graphviz)

### 1. Clone/Download the Project
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install Dependencies
```bash
pip install matplotlib tqdm graphviz
```

### 3. Prepare Your Data
- Place your CSV data in the `dataset/` folder (see example in code)

### 4. Train the Model
Run the training script (e.g. `demo.py`):
```bash
python demo.py
```
This will:
- Load and preprocess your data
- Train a neural network from scratch
- Print training and validation errors
- Output a sample prediction

### 5. Visualize the Computation Graph (Optional)
In your script, use:
```python
import grapher
# After computing an output, e.g. result = model(sample_input)
grapher.draw_dot(result)
```

## 📁 Project Structure
```
project-root/
├── demo.py                # Training and prediction script
├── neuron.py              # Core neural network/math classes
├── grapher.py             # Computation graph visualizer
├── predict.py             # (Optional) Prediction script
├── dataset/               # Folder for CSV data
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🎨 Features

- **No external ML libraries**: 100% pure Python math
- **Customizable architecture**: Change layers, activations, etc.
- **Educational**: Transparent, step-by-step implementation
- **Computation graph visualization**: See how data and gradients flow
- **Progress bars and error plots**: Track training visually

## 🔧 Customization

- **Model Architecture**: Edit `demo.py` to change layer sizes, activations, epochs, or learning rate
- **Data**: Use your own CSVs (just match the input format)
- **Visualization**: Use `grapher.py` to visualize any computation

## 🐛 Troubleshooting

- **Model always predicts 0**: Check your data, learning rate, and training epochs
- **Import errors**: Ensure all dependencies are installed
- **Graphviz errors**: Install Graphviz system package if needed

## 📊 Model Performance
- Performance depends on your data and chosen architecture
- This project is for learning, not for state-of-the-art results

## 🤝 Contributing
Pull requests and suggestions are welcome! This project is meant for learning and experimentation.

## 📄 License
This project is open source and available under the MIT License. 