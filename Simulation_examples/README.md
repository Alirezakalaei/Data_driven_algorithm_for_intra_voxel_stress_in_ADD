# Dislocation Dynamics Simulations: The Role of Intra-Voxel Stress

This folder contains application examples of the trained CNN-MLP hybrid model. These scripts run continuum dislocation dynamics simulations for FCC Copper, explicitly resolving **intra-voxel stress** fields using the neural network predictor. 

In many standard coarse-grained simulations, intra-voxel (sub-grid) stress fluctuations are neglected, which can lead to inaccuracies in predicting dislocation nucleation, annihilation, and yielding behavior. The simulations provided here are designed to demonstrate the critical role of these intra-voxel stresses under various macroscopic boundary conditions, system sizes, and nucleation source densities, as discussed in the thesis.

## 📁 File Structure

* **`constant_stress.py`**: Runs a creep simulation under a constant applied macroscopic stress. It tracks the evolution of dislocation density, strain, and average velocity over time.
* **`constant_strain.py`**: Runs a constant strain rate simulation. It uses a damped Proportional (P-Control) feedback loop to dynamically adjust the applied stress to maintain a target macroscopic strain rate.
* **`computational_functions.py`**: A core library of Numba-accelerated numerical methods (Finite Difference/Volume Methods, Upwind schemes, MUSCL) for solving the dislocation transport equations, annihilation, and nucleation.
* **`cubic_functions.py`**: Contains geometric and physics functions tailored for Cartesian/cubic simulation grids, including continuous density coarse-graining and Mura's stress kernel computations.

## ⚙️ Prerequisites

To run these simulations, you must have the trained neural network files from the ML pipeline in the same directory (or update the paths in the scripts):
1. `packaged_hybrid_model_ensemble_0.pth` to `..._4.pth` (The ensemble model weights).
2. `best_hyperparameters.json` (The architecture configuration).
3. `nn_predictor.py` (The helper script that loads the PyTorch models and Scikit-Learn scalers).

## 🚀 Usage

### 1. Install Dependencies
Ensure your Python environment has the required packages installed:
```bash
pip install -r requirements.txt
