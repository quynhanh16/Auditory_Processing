# Pure Tone Neural Analysis

## Introduction

The Pure Tone Neural Analysis project is designed to analyze neural responses to auditory stimuli, specifically pure tones. The project processes neural data, computes firing rates, and visualizes the relationship between auditory stimuli and neural activity.

## Before Getting Started

Ensure you have Python version 3.9 or higher installed on your system.

### Steps to Set Up the Environment

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo/PureToneNeuralAnalysis.git
   ```

2. **Create a Python Virtual Environment**

   ```bash
   python -m venv ./pure-tone-env
   ```

3. **Activate the Virtual Environment**

   On macOS/Linux:

   ```bash
   source pure-tone-env/bin/activate
   ```

   On Windows:

   ```bash
   .\pure-tone-env\Scripts\activate
   ```

4. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

### Downloading the Data

The project requires neural response data and auditory stimuli files. Ensure the following files are downloaded and placed in the appropriate directories:

- Neural response data: Place `.mat` files in the `./data/pure_tones_spikes/` directory.
- Stimulus data: Place `PureToneSound.wav` in the `./data/Stimulus/` directory.

## Running the Project

After setting up the environment and downloading the data, you can run the project by executing the `main.py` script. This script includes functions for processing data, training models, and generating visualizations.

```bash
python main.py
```

## Project Structure

The project is organized into the following directories and files:

```
data/
tools/
    firing_rate.py
    gammatone.py
    graphing.py
    normalization.py
    utils.py
main.py
requirements.txt
README.md
```

- **`data/`**: Contains the raw and processed data files.
- **`tools/`**: Includes utility scripts for data processing, firing rate calculations, normalization, and visualization.
- **`main.py`**: The main script for running the project.
- **`requirements.txt`**: Lists the dependencies required for the project.

## Key Functions

### Data Loading and Processing

- **`load_data()`**: Loads neural response data and aligns it with stimulus triggers.
- **`load_stimuli()`**: Processes auditory stimuli into a spectrogram-like representation using gammatone filters.

### Firing Rate Analysis

- **`firing_rate()`**: Computes firing rates for individual neurons or populations using fixed or sliding windows.

### Visualization

- **`fra_plot()`**: Generates Frequency Response Area (FRA) plots for neurons.
- **`stimuli_heatmap()`**: Creates heatmaps to visualize the relationship between stimuli and neural responses.

### Machine Learning

- **`run_linear_model()`**: Prepares stimuli and response data, trains a linear regression model, and evaluates its performance.

## Results

The project outputs various results, including:

- **Firing rate plots**: Visualize the firing rate over time.
- **Heatmaps**: Show the relationship between stimuli and neural responses.
- **Model coefficients**: Save the trained model's coefficients for further analysis.

## Credit

This project is inspired by auditory neuroscience research. If you use this project in your work, please cite the relevant papers and repositories. For questions or issues, open an issue on the GitHub repository.