from typing import Union
import torch


def rigid_transform_3D(target_face: torch.tensor, mean_face: torch.tensor) -> torch.tensor:
    """
        Compute a rigid transformation between two sets of landmarks by using Kabsch algorithm.
        The Kabsch algorithm, named after Wolfgang Kabsch, is a method for calculating the optimal rotation matrix
        that minimizes the RMSD (root mean squared deviation) between two paired sets of points.
        args:
            target_face: NumPy array of shape (3,N)
            mean_face: NumPy array of shape (3,N)

        returns:
            R: NumPy array of shape (3,3)
            t: NumPy array of shape (3,1)

        source:
            https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    # Geometric transformations in 3D
    # https://cseweb.ucsd.edu/classes/wi18/cse167-a/lec3.pdf

    # Affine transformation (theoretical)
    # http://learning.aols.org/aols/3D_Affine_Coordinate_Transformations.pdf

    # Implementation from http://nghiaho.com/?page_id=671
    #
    assert target_face.shape == mean_face.shape
    assert target_face.shape[0] == 3, "3D rigid transform only"

    # find mean column wise
    centroid_A = torch.mean(target_face, dim=1)
    centroid_B = torch.mean(mean_face, dim=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = target_face - centroid_A
    Bm = mean_face - centroid_B

    H = Am @ Bm.T
    # H = (Am.cpu() @ Bm.T.cpu())

    # find rotation
    U, S, Vt = torch.linalg.svd(H) # torch.svd differs from torch.linalg.svd
    # https://pytorch.org/docs/stable/generated/torch.svd.html
    R = Vt.T @ U.T

    # special reflection case
    if torch.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def apply_transformation(R, t, landmarks: torch.tensor) -> torch.tensor:
    """
        Apply a rigid transformation to a set of landmarks.
        args:
            R: NumPy array of shape (3,3)
            t: NumPy array of shape (3,1)
            landmarks: NumPy array of shape (3,N)
    """
    assert landmarks.shape[0] == 3, "landmarks must be 3D"
    assert R.shape == (3, 3), "R must be 3x3"
    assert t.shape == (3, 1), "t must be 3x1"

    # apply transformation
    transformed_landmarks = R @ landmarks + t

    return transformed_landmarks


def register_sequence_of_landmarks(target_sequence: torch.tensor, mean_face: torch.tensor, per_frame=False,
                                   display_sequence: Union[torch.tensor, None] = None) -> torch.tensor:
    """
        Register a sequence of landmarks to a mean face.
        Computational complexity: O(3*N*T)
        args:
            target_face: NumPy array of shape (T,3,N)
            mean_face: NumPy array of shape (3,N)
            per_frame: either to estimate the transformation per frame or given the mean face.
            display_sequence: (optional) NumPy array of shape (T',3,N'). Optional array to estimate the transformation
            on some of the landmarks.

        returns:
            registered_sequence: NumPy array of shape (T,3,N)

        example:
            Computing the transformation ignoring landmarks from 48 onwards but
            estimating the transformation for all of them
            >>> registered_sequence = register_sequence_of_landmarks(landmarks[..., :48],
            >>>                                                     mean_face[:, :48],
            >>>                                                     display_sequence=landmarks)
    """
    if display_sequence is None:
        display_sequence = target_sequence

    if not per_frame:
        # Estimates the mean face
        target_mean_face = torch.mean(target_sequence, dim=0)
        # compute rigid transformation
        R, t = rigid_transform_3D(target_mean_face, mean_face)

    # apply transformation
    registered_sequence = []
    for x, y in zip(target_sequence, display_sequence):
        if per_frame:
            R, t = rigid_transform_3D(x, mean_face)
        registered_sequence.append(apply_transformation(R, t, y))

    return torch.stack(registered_sequence)
