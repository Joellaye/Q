import numpy as np

def perform_pca(returns):
    cov_matrix = returns.cov()
    eigenvalues = np.linalg.eig(cov_matrix)[0]
    eigenvectors = np.linalg.eig(cov_matrix)[1]
    # Sort eigenvalues and eigenvectors

    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:", eigenvectors)

