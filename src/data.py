from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import make_blobs, make_moons
from numpy.lib.stride_tricks import sliding_window_view

@dataclass
class Dataset:
    """
    Dataset represented by a feature matrix X and optional labels y.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    y : np.ndarray | None
        Optional label vector of shape (n_samples,).
    """

    X: np.ndarray
    y: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(
                "X must have shape (n_samples, n_features)."
            )

        if self.y is not None:
            if self.y.ndim != 1:
                raise ValueError(
                    "y must be a one-dimensional array."
                )

            if len(self.y) != len(self.X):
                raise ValueError(
                    "X and y must contain the same number of samples."
                )
    
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]
    
    @property 
    def n_features(self) -> int:
        return self.X.shape[1]


@dataclass
class SeedSet:
    """
    Set of labelled graph vertices.

    Parameters
    ----------
    indices : np.ndarray
        Vertex indices of shape (n_seeds,).

    labels : np.ndarray
        Corresponding class labels of shape (n_seeds,).
    """

    indices: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.indices.ndim != 1:
            raise ValueError(
                "indices must be a one-dimensional array."
            )

        if self.labels.ndim != 1:
            raise ValueError(
                "labels must be a one-dimensional array."
            )

        if len(self.indices) != len(self.labels):
            raise ValueError(
                "indices and labels must have the same length."
            )

        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError(
                "Seed indices must be unique."
            )
    
    @property
    def n_seeds(self) -> int:
        return len(self.indices)

def make_two_moons(
    n_samples: int = 500,
    noise: float = 0.08,
    random_state: int | None = None,
) -> Dataset:
    """
    Generate the classical two-moons classification dataset.
    """
    
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )

    return Dataset(X=X, y=y)

def make_point_cloud(
    n_samples: int = 500,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int | None = None,
) -> Dataset:
    """
    Generate a synthetic point cloud composed of Gaussian clusters.
    """

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    return Dataset(X=X, y=y)

def select_seeds(
    y: np.ndarray,
    n_per_class: int = 1,
    random_state: int | None = None,
) -> SeedSet:
    """
    Randomly select the same number of labelled seeds
    from each class.
    """

    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    selected_indices = []
    classes = np.unique(y)

    for label in classes:
        class_indices = np.flatnonzero(y == label)

        chosen = rng.choice(
            class_indices,
            size=n_per_class,
            replace=False,
        )

        selected_indices.extend(chosen.tolist())
    
    indices = np.asarray(
        selected_indices,
        dtype=int,
    )
    labels = y[indices]

    return SeedSet(
        indices=indices,
        labels=labels,
    )


def load_image(
    path : str | Path,
    grayscale: bool = False,
) -> np.ndarray:
    """
    Load an image and return normalized values in [0, 1].
    """
    path = Path(path)
    
    if not path.existe():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )
    
    with Image.open(path) as image:

        if grayscale:
            image = image.convert("L")
        else :
            image = image.convert("RGB")
        
        array = np.assarray(
            image,
            dtype=np.float64,
        )

    return array / 255.0

def load_dataset(
    path: str | Path,
) -> Dataset:
    """
    Load a dataset stored in NumPy .npz format.

    The file must contain an array named 'X'.
    The array 'y' is optional.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )
    
    if path.suffix != ".npz":
        raise ValueError(
            "Only .npz datasets are currently supported."
        )
    
    with np.load(path) as data:

        if "X" not in data:
            raise ValueError(
                "The dataset file must contain "
                "an array named 'X'."
            )
        
        X = data["X"]
        y = data["y"] if y in data else None
    
    return Dataset(
        X=X,
        y=y
    )

def extract_patches(
    image: np.ndarray,
    patch_size: int = 5,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """
    Extract one square patch centered at each image pixel.

    The patches are flattened into feature vectors.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W) or
        color image of shape (H, W, C).

    patch_size : int
        Odd patch width.

    padding_mode : str
        NumPy padding mode used at image boundaries.

    Returns
    -------
    np.ndarray
        Patch feature matrix of shape

            (H * W, patch_dimension).
    """

    image = np.asarray(
        image,
        dtype=float,
    )

    if image.ndim not in (2, 3):
        raise ValueError(
            "image must have shape (H, W) "
            "or (H, W, C)."
        )

    if (
        patch_size < 1
        or patch_size % 2 == 0
    ):
        raise ValueError(
            "patch_size must be a positive odd integer."
        )

    height, width = image.shape[:2]

    radius = patch_size // 2

    if image.ndim == 2:

        padded = np.pad(
            image,
            (
                (radius, radius),
                (radius, radius),
            ),
            mode=padding_mode,
        )

        windows = sliding_window_view(
            padded,
            (
                patch_size,
                patch_size,
            ),
        )

        return windows.reshape(
            height * width,
            patch_size * patch_size,
        )

    channels = image.shape[2]

    padded = np.pad(
        image,
        (
            (radius, radius),
            (radius, radius),
            (0, 0),
        ),
        mode=padding_mode,
    )

    windows = sliding_window_view(
        padded,
        (
            patch_size,
            patch_size,
        ),
        axis=(0, 1),
    )

    # sliding_window_view returns:
    #
    # (H, W, C, patch_size, patch_size)
    #
    # Reorder into:
    #
    # (H, W, patch_size, patch_size, C)

    windows = windows.transpose(
        0,
        1,
        3,
        4,
        2,
    )

    return windows.reshape(
        height * width,
        patch_size
        * patch_size
        * channels,
    )