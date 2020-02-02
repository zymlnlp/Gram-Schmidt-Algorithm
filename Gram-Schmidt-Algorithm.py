import numpy as np

A = np.random.rand(5,5) #randomly generate a matrix for testing

def gram_schmidt(A):
    
    A = A.astype(np.float64)

    orthogonal_basis = []

    for column_count in range(A.shape[1]):

        current_column_vector = A[:,column_count]

        if column_count == 0:
            orthogonal_basis.append(current_column_vector)
            continue

        for ortho_count in range(len(orthogonal_basis)+1):

            if ortho_count == 0:
                current_orthogonal = current_column_vector
                continue

            previous_orthogonal = orthogonal_basis[ortho_count - 1]
            current_orthogonal = current_orthogonal - np.dot(((np.dot(current_column_vector.T, previous_orthogonal)) / (np.dot(previous_orthogonal.T, previous_orthogonal))), previous_orthogonal)

        orthogonal_basis.append(current_orthogonal)

    orthonormal_basis = []
    
    for num in orthogonal_basis:
        if np.linalg.norm(num) == 0:
            continue
        orthonormal_basis.append((num / np.linalg.norm(num)))
    
    orthonormal_basis = np.array(orthonormal_basis).T
    return orthonormal_basis
    
print(gram_schmidt(A))
