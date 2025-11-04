# Auditory Processing

## Introduction

The Pure Tone Neural Analysis project is designed to analyze neural responses to auditory stimuli, specifically pure tones. The project processes neural data, computes firing rates, and visualizes the relationship between auditory stimuli and neural activity.

## Before Getting Started

Ensure you have Python version 3.9 or higher installed on your system.

### Steps to Set Up the Environment

1. **Clone the Repository**

   ```bash
   git clone https://github.com/quynhanh16/Auditory_Processing.git
   ```

2. **Create a Python Virtual Environment**

   ```bash
   python -m venv ./.venv
   ```

3. **Activate the Virtual Environment**

   On macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```bash
   .\.venv\Scripts\activate
   ```

4. **Install Required Packages**

   ```bash
   pip install .
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

## Credit

This project is inspired by auditory neuroscience research. If you use this project in your work, please cite the relevant papers and repositories. For questions or issues, open an issue on the GitHub repository.