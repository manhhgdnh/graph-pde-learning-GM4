from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import make_blobs, make_moons
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)

        if self.X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")

        if min(self.X.shape) == 0:
            raise ValueError("X must not be empty.")

        if not np.all(np.isfinite(self.X)):
            raise ValueError("X must contain finite values.")

        if self.y is not None:
            self.y = np.asarray(self.y)

            if self.y.shape != (self.X.shape[0],):
                raise ValueError("y must have shape (n_samples,).")

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


@dataclass
class SeedSet:
    indices: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices)
        self.labels = np.asarray(self.labels)

        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional.")

        if self.indices.size == 0:
            self.indices = self.indices.astype(int)

        if self.indices.dtype.kind not in "iu":
            raise TypeError("Seed indices must be integers.")

        self.indices = self.indices.astype(int, copy=True)

        if self.labels.shape != self.indices.shape:
            raise ValueError("indices and labels must have the same shape.")

        if np.unique(self.indices).size != self.indices.size:
            raise ValueError("Seed indices must be unique.")

        if self.labels.dtype.kind in "fc":
            if not np.all(np.isfinite(self.labels)):
                raise ValueError("Seed labels must be finite.")

    @property
    def n_seeds(self) -> int:
        return self.indices.size


def make_two_moons(
    n_samples: int = 500,
    noise: float = 0.08,
    random_state: int | None = None,
) -> Dataset:
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )
    return Dataset(X, y)


def make_point_cloud(
    n_samples: int = 500,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int | None = None,
) -> Dataset:
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return Dataset(X, y)


def select_seeds(
    y: np.ndarray,
    n_per_class: int = 1,
    random_state: int | None = None,
) -> SeedSet:
    y = np.asarray(y)

    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array.")

    if not isinstance(n_per_class, (int, np.integer)):
        raise TypeError("n_per_class must be an integer.")

    if n_per_class < 1:
        raise ValueError("n_per_class must be positive.")

    rng = np.random.default_rng(random_state)
    selected = []

    for label in np.unique(y):
        candidates = np.flatnonzero(y == label)

        if candidates.size < n_per_class:
            raise ValueError("A class contains fewer points than requested.")

        chosen = rng.choice(
            candidates,
            size=n_per_class,
            replace=False,
        )
        selected.extend(chosen.tolist())

    indices = np.asarray(selected, dtype=int)
    return SeedSet(indices, y[indices])


def load_image(
    path: str | Path,
    grayscale: bool = False,
) -> np.ndarray:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        mode = "L" if grayscale else "RGB"
        array = np.asarray(image.convert(mode), dtype=float)

    return array / 255.0


def load_dataset(path: str | Path) -> Dataset:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix != ".npz":
        raise ValueError("Only .npz datasets are supported.")

    with np.load(path, allow_pickle=False) as data:
        if "X" not in data:
            raise ValueError("The dataset must contain an array named X.")

        X = data["X"].copy()
        y = data["y"].copy() if "y" in data else None

    return Dataset(X, y)


def extract_patches(
    image: np.ndarray,
    patch_size: int = 5,
    padding_mode: str = "reflect",
) -> np.ndarray:
    image = np.asarray(image, dtype=float)

    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or color image.")

    if not np.all(np.isfinite(image)):
        raise ValueError("image must contain finite values.")

    if not isinstance(patch_size, (int, np.integer)):
        raise TypeError("patch_size must be an integer.")

    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")

    if image.ndim == 2:
        image = image[:, :, None]

    height, width, channels = image.shape
    radius = patch_size // 2

    padded = np.pad(
        image,
        ((radius, radius), (radius, radius), (0, 0)),
        mode=padding_mode,
    )

    windows = sliding_window_view(
        padded,
        (patch_size, patch_size),
        axis=(0, 1),
    )

    windows = windows.transpose(0, 1, 3, 4, 2)

    return windows.reshape(
        height * width,
        patch_size * patch_size * channels,
    )

