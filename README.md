# Auditory_Processing

## Introduction

The Auditory_Processing project aims to leverage Convolutional Neural Networks (CNNs) to simulate auditory systems using
data from the [Zenodo repository](https://zenodo.org/records/8044773). This project encompasses various stages,
including data loading, analysis, model training, evaluation, and result presentation.

## Before Getting Started

Due to the developmental state of the NEMS package used in this project, there are some preliminary steps required
before running the program. Ensure you have Python version 3.9 or 3.10 installed on your system.

### Steps to Set Up the Environment

1. **Clone the Auditory_Processing Repository**

   ```bash
   git clone https://github.com/JonathanJT109/Auditory_Processing
   ```

2. **Clone the NEMS Repository**

   ```bash
   git clone https://github.com/LBHB/NEMS.git
   ```

3. **Create a Python Virtual Environment**

   ```bash
   python -m venv ./nems-env
   ```

4. **Activate the Virtual Environment**

   On Windows:

   ```bash
   .\nems-env\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   source nems-env/bin/activate
   ```

5. **Install NEMS and Required Packages**

   ```bash
   pip install NEMS
   ```

6. **Rename the NEMS Directory**

   Rename the directory named `NEMS` to `NEMS_` (or any other name you prefer).

7. **Move the `nems` Directory**

   Move the `nems` directory located inside the newly renamed `NEMS_` directory to your home directory.

### Downloading the data

The dataset used in this project can be located at [Zenodo repository](https://zenodo.org/records/8044773). The file
used in this project is: **A1_NAT4_ozgf.fs100.ch18.tgz**. After downloading the file, place the tgz file in the home
directory. Now, you are ready.

## Running the Project

After setting up the environment and preparing the directories, you should be able to run the files included in the
Auditory_Processing project.

## Credit

Please cite the following work if you use the data from this project in your publications:

- [Biological Cybernetics Paper](https://www.biorxiv.org/content/10.1101/2022.06.10.495698v2)

For any questions or issues, please open an issue on
the [GitHub repository](https://github.com/JonathanJT109/Auditory_Processing).
