# Data-Driven Algorithm for Intra-Voxel Stress in CDD

This repository contains the source code developed as part of my thesis for computing stress loss during coarse graining in Continuum Dislocation Dynamics (CDD). 

The project introduces a machine learning approach using a Convolutional Neural Network combined with a Multi-Layer Perceptron (CNN-MLP). The repository includes scripts for generating dislocation microstructures, training the CNN-MLP algorithm, and applying the trained model to simulate dislocation plasticity under constant stress and constant strain conditions.

## Repository Structure

The repository is divided into two main directories, reflecting the two core stages of the methodology:

### 1. `Training_CNN_MLP`
This directory contains the code required to build and train the machine learning model. 
* **Purpose:** Generates dislocation microstructures, processes the data, and trains the CNN-MLP architecture to predict intra-voxel stress.
* **Output:** Running the scripts in this folder will generate the trained model weights and data required for the subsequent CDD simulations.

### 2. `Simulation_examples`
This directory contains the scripts for running Continuum Dislocation Dynamics (CDD) simulations.
* **Purpose:** Applies the trained CNN-MLP model to simulate dislocation plasticity. It includes specific simulation scenarios, such as deformation under **constant stress** and **constant strain**.
* **Input:** These simulations require the trained model data generated from the `Training_CNN_MLP` step.

## Workflow: How to Use This Repository

To reproduce the results or utilize the framework, please follow this workflow:

1. **Train the Model:** 
   Navigate to the `Training_CNN_MLP` directory and run the training scripts. This will train the CNN-MLP algorithm and output the necessary model files/weights.
   
2. **Transfer the Trained Model:** 
   Once training is complete, ensure the resulting model data/weights are placed in the appropriate location within the `Simulation_examples` directory (as specified in the simulation scripts).

3. **Run CDD Simulations:** 
   Navigate to the `Simulation_examples` directory. You can now run the CDD simulation scripts for either constant stress or constant strain, which will utilize the trained CNN-MLP model to compute stress loss during the simulation.

## Requirements

The required Python packages to run this code are listed in the `requirements.txt` file. You can install all dependencies using pip:

```bash
pip install -r requirements.txt
