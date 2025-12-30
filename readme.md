- Python code to simulate Ising model
- See file for `Ising_model_mc_simulation.ipynb'  for more  details of simulation
- MPI installion
"
pip install mpi4py
python3 -m pip install mpi4py
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin
brew install open-mpi
python3 -m pip show mpi4py
mpiexec --version
mpiexec -n 4 python Ising_MPI.py
"
