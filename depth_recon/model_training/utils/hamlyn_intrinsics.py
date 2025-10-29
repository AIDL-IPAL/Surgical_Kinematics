import numpy as np

def read_intrinsics(path):
    """
    Reads a 3×4 camera matrix from `path` and returns [fx, fy, cx, cy].
    Assumes the file is plain text with 3 rows of 4 numbers each.
    """
    # load the full 3x4 matrix
    M = np.loadtxt(path)       # shape (3,4)

    # fx = M[0,0], fy = M[1,1], cx = M[0,2], cy = M[1,2]
    fx = M[0, 0]
    fy = M[1, 1]
    cx = M[0, 2]
    cy = M[1, 2]

    return np.array([fx, fy, cx, cy], dtype=float)

# Example usage:
if __name__ == "__main__":
    intr = read_intrinsics("intrinsics.txt")
    print("Flattened intrinsics [fx, fy, cx, cy]:", intr)
