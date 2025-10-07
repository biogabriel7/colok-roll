"""Static plotting utilities for microscopy data."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from ..config import ProjectionConfig


class Visualizer:
    """Basic visualization tools for microscopy images."""

    def __init__(self, config: Optional[ProjectionConfig] = None):
        self.config = config or ProjectionConfig()
        plt.rcParams["figure.dpi"] = self.config.figure_dpi

    def plot_image(
        self,
        image: np.ndarray,
        title: Optional[str] = None,
        colormap: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8),
        show_colorbar: bool = True,
        ax: Optional[Axes] = None,
    ) -> Tuple[Figure, Axes]:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        colormap = colormap or self.config.default_colormap

        if image.ndim == 2:
            im = ax.imshow(image, cmap=colormap)
            if show_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            if image.max() > 1:
                image = image / image.max()
            ax.imshow(image)
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

        ax.set_title(title or "Image")
        ax.axis("off")
        return fig, ax

    def plot_channels(
        self,
        image: np.ndarray,
        channel_names: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        colormap: str = "gray",
    ) -> Figure:
        if image.ndim == 4:
            image = np.max(image, axis=0)
        elif image.ndim != 3:
            raise ValueError(f"Image must be 3D or 4D, got {image.ndim}D")

        n_channels = image.shape[2]
        channel_names = channel_names or [f"Channel {i}" for i in range(n_channels)]
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols
        figsize = figsize or (n_cols * 4, n_rows * 4)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for i in range(n_channels):
            ax = axes[i]
            channel_data = image[:, :, i]
            im = ax.imshow(channel_data, cmap=colormap)
            ax.set_title(channel_names[i])
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(n_channels, len(axes)):
            axes[i].axis("off")

        plt.suptitle("Individual Channels", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_composite(
        self,
        image: np.ndarray,
        channel_colors: Optional[Dict[int, str]] = None,
        channel_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 10),
        show_individual: bool = True,
    ) -> Figure:
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D (Y, X, C), got {image.ndim}D")

        n_channels = image.shape[2]
        default_colors = {
            0: "blue",
            1: "green",
            2: "red",
            3: "cyan",
            4: "magenta",
            5: "yellow",
        }
        channel_colors = channel_colors or {i: default_colors.get(i, "gray") for i in range(n_channels)}
        channel_names = channel_names or [f"Channel {i}" for i in range(n_channels)]

        composite = _create_composite_image(image, channel_colors)

        if show_individual:
            n_cols = n_channels
            fig, axes = plt.subplots(2, n_cols, figsize=figsize)
            ax_composite = axes[0] if n_cols == 1 else axes[0, :]
            if np.ndim(ax_composite) > 1:
                ax_main = ax_composite[0]
            else:
                ax_main = ax_composite
            ax_main.imshow(composite)
            ax_main.set_title("Composite Image")
            ax_main.axis("off")

            individual_axes = axes[1] if n_cols > 1 else [axes[1]]
            for i in range(n_channels):
                ax = individual_axes[i]
                color = channel_colors.get(i, "gray")
                cmap = _get_color_colormap(color)
                ax.imshow(image[:, :, i], cmap=cmap)
                ax.set_title(channel_names[i], fontsize=10)
                ax.axis("off")

            for j in range(n_channels, len(individual_axes)):
                individual_axes[j].axis("off")

            plt.tight_layout()
            return fig

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(composite)
        ax.set_title("Composite Image")
        ax.axis("off")
        return fig

    def plot_intensity_histogram(
        self,
        image: np.ndarray,
        channel_names: Optional[List[str]] = None,
        bins: int = 256,
        figsize: Tuple[int, int] = (10, 6),
        log_scale: bool = True,
    ) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        n_channels = image.shape[2]
        channel_names = channel_names or [f"Channel {i}" for i in range(n_channels)]
        colors = ["blue", "green", "red", "cyan", "magenta", "yellow"]

        for c in range(n_channels):
            channel_data = image[:, :, c].flatten()
            channel_data = channel_data[channel_data > 0]
            if channel_data.size == 0:
                continue
            color = colors[c % len(colors)]
            ax.hist(
                channel_data,
                bins=bins,
                alpha=0.5,
                label=channel_names[c],
                color=color,
                density=True,
                histtype="stepfilled",
            )

        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency (normalized)")
        ax.set_title("Intensity Distribution")
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_z_profile(
        self,
        z_stack: np.ndarray,
        position: Optional[Tuple[int, int]] = None,
        channel: int = 0,
        figsize: Tuple[int, int] = (10, 6),
    ) -> Figure:
        if z_stack.ndim == 3:
            z_stack = z_stack[:, :, :, np.newaxis]
        n_slices, height, width, _ = z_stack.shape
        position = position or (height // 2, width // 2)
        y, x = position

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        z_profile = z_stack[:, y, x, channel]
        ax1.plot(range(n_slices), z_profile, "b-", linewidth=2)
        ax1.set_xlabel("Z-slice")
        ax1.set_ylabel("Intensity")
        ax1.set_title(f"Z-profile at position ({y}, {x})")
        ax1.grid(True, alpha=0.3)

        mip = np.max(z_stack[:, :, :, channel], axis=0)
        ax2.imshow(mip, cmap="gray")
        ax2.plot(x, y, "r+", markersize=15, markeredgewidth=2)
        ax2.set_title("Position on MIP")
        ax2.axis("off")

        plt.tight_layout()
        return fig


def plot_mip(
    mip: np.ndarray,
    title: str = "Maximum Intensity Projection",
    colormap: str = "viridis",
    figsize: Tuple[int, int] = (8, 8),
) -> Figure:
    visualizer = Visualizer()
    fig, _ = visualizer.plot_image(mip, title=title, colormap=colormap, figsize=figsize)
    return fig


def plot_channels(image: np.ndarray, channel_names: Optional[List[str]] = None) -> Figure:
    visualizer = Visualizer()
    return visualizer.plot_channels(image, channel_names)


def _create_composite_image(image: np.ndarray, channel_colors: Dict[int, str]) -> np.ndarray:
    height, width, n_channels = image.shape
    composite = np.zeros((height, width, 3), dtype=np.float32)
    color_map = {
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
        "yellow": (1, 1, 0),
        "gray": (1, 1, 1),
        "white": (1, 1, 1),
    }

    for c in range(n_channels):
        color_name = channel_colors.get(c, "gray")
        if color_name in color_map:
            color = color_map[color_name]
        else:
            try:
                import matplotlib.colors as mcolors

                color = mcolors.to_rgb(color_name)
            except Exception:
                color = (1, 1, 1)

        channel_data = image[:, :, c].astype(np.float32)
        if channel_data.max() > 0:
            channel_data = channel_data / channel_data.max()
        for i in range(3):
            composite[:, :, i] += channel_data * color[i]

    return np.clip(composite, 0, 1)


def _get_color_colormap(color: str) -> str:
    colormaps = {
        "red": "Reds",
        "green": "Greens",
        "blue": "Blues",
        "cyan": "GnBu",
        "magenta": "RdPu",
        "yellow": "YlOrBr",
    }
    return colormaps.get(color, "gray")


