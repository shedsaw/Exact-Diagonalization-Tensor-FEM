# Exact-Diagonalization-Tensor-FEM
Numerical algorithm implementing the exact diagonalization technique for a tensor FEM for solving the spectral fractional Laplacian.

This implementation requires an installation of the deal.ii finite element library.
Assuming deal.ii has been installed and properly configured, the following should build the application:

user@linux:~/Exact-Diagonalization-Tenser-FEM$ cmake .
user@linux:~/Exact-Diagonalization-Tenser-FEM$ make release
user@linux:~/Exact-Diagonalization-Tenser-FEM$ make all

The "spectral-fractional-laplacian.prm" may be edited to select which geometry and source term is to be used. The fractional power s (0 < s < 1) may be
adjusted.

To run the code,

user@linux:~/Exact-Diagonalization-Tenser-FEM$ mpirun -np $NP ./spectral-fractional-laplacian