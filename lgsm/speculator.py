"""This module defines the spectral decoder that predicts spectra."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import nn

from .paths import paths


@dataclass
class SpeculatorConfig:
    n_latents: int
    n_hidden: list
    n_pcas: int
    n_wavelengths: int
    wavelength_min: float
    wavelength_max: float


speculator_uv_config = SpeculatorConfig(
    n_latents=14,
    n_hidden=[256, 256, 256, 256],
    n_pcas=50,
    n_wavelengths=673,
    wavelength_min=1_005,
    wavelength_max=4_000,
)

speculator_optical_config = SpeculatorConfig(
    n_latents=14,
    n_hidden=[256, 256, 256],
    n_pcas=30,
    n_wavelengths=3929,
    wavelength_min=4_000,
    wavelength_max=11_000,
)


class SpeculatorActivation(nn.Module):
    """Custom activation function from Alsing 2020 (Eq.8).

    Note there is a variable change from the paper:
    beta -> alpha
    gamma -> beta
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.rand([size]))
        self.beta = nn.Parameter(torch.rand([size]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function.

        Parameters
        ----------
        x: torch.Tensor
            Data to which the activation is applied

        Returns
        -------
        torch.tensor
            Data with activation applied
        """
        return (self.beta + torch.sigmoid(self.alpha * x) * (1 - self.beta)) * x


class SpeculatorModule(nn.Module):
    def __init__(self, config: SpeculatorConfig) -> None:
        """Create the Speculator module."""
        super().__init__()

        # Save config
        self.config = config

        # Pull out some useful info
        n_nodes = [self.config.n_latents] + self.config.n_hidden + [self.config.n_pcas]
        n_layers = len(n_nodes) - 1

        # Create random weights and biases...

        # Latents
        self.latent_shift = nn.Parameter(torch.rand(self.config.n_latents))
        self.latent_scale = nn.Parameter(torch.rand(self.config.n_latents))

        # Linear layers
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(n_nodes[i], n_nodes[i + 1]))
            layers.append(SpeculatorActivation(n_nodes[i + 1]))
        layers.append(nn.Linear(n_nodes[-2], n_nodes[-1]))
        self.linear_layers = nn.Sequential(*layers)

        # PCA coefficients
        self.pca_shift = nn.Parameter(torch.rand(self.config.n_pcas))
        self.pca_scale = nn.Parameter(torch.rand(self.config.n_pcas))

        # Spectrum reconstruction
        self.wavelengths = nn.Parameter(
            torch.linspace(
                config.wavelength_min,
                config.wavelength_max,
                config.n_wavelengths,
            )
        )
        self.pca_basis = nn.Parameter(
            torch.rand([self.config.n_pcas, self.config.n_wavelengths])
        )
        self.log_spectrum_shift = nn.Parameter(torch.rand(self.config.n_wavelengths))
        self.log_spectrum_scale = nn.Parameter(torch.rand(self.config.n_wavelengths))

        # Freeze most parameters
        self.freeze()

    def predict_pca_coeff(self, latents: torch.Tensor) -> torch.tensor:
        """Predict PCA coefficients for latent parameters."""
        # Normalize latent variables
        x = (latents - self.latent_shift) / self.latent_scale

        # Pass through linear layers
        x = self.linear_layers(x)

        # Rescale the PCA coefficients
        pca_coeff = (x * self.pca_scale) + self.pca_shift

        return pca_coeff

    def reconstruct_spectrum(
        self,
        pca_coeff: torch.tensor,
        log_mass: torch.tensor,
    ) -> torch.tensor:
        """Reconstruct the SED from the PCA coefficients."""
        # Linear combo of basis vector
        log_spectrum = pca_coeff @ self.pca_basis

        # Undo normalization
        log_spectrum = log_spectrum * self.log_spectrum_scale + self.log_spectrum_shift

        # Add log of stellar mass: log10(stellar mass)
        log_spectrum = log_spectrum + log_mass * math.log(10)

        # Undo log
        spectrum = torch.exp(log_spectrum)

        return spectrum

    def forward(self, latents: torch.Tensor, log_mass: torch.Tensor) -> torch.tensor:
        """Map latents onto an SED."""
        return self.reconstruct_spectrum(self.predict_pca_coeff(latents), log_mass)

    def freeze(
        self,
        latents: bool = True,
        linear: bool = True,
        activation: bool = True,
        pca_norm: bool = True,
        pca_basis: bool = True,
        wavelengths: bool = True,
        spectrum: bool = True,
    ) -> None:
        """Freeze network parameters during training.

        Parameters
        ----------
        latents: bool, default=True
            Whether to freeze the parameters that normalize the latent variables.
        linear: bool, default=True
            Whether to freeze the weights in the linear layers
        activation: bool, default=True
            Whether to freeze the parameters in the activation layers
        pca_norm: bool, default=True
            Whether to freeze the parameters that normalize the PCA coefficients
        pca_basis: bool, default=True
            Whether to freeze the PCA basis vectors
        wavelengths: bool, default=True
            Whether to freeze the values in the wavelength grid
        spectrum: bool, default=True
            Whether to freeze the parameters that normalize the spectrum
        """
        if latents:
            self.latent_shift.requires_grad = False
            self.latent_scale.requires_grad = False
        if linear:
            for layer in self.linear_layers[::2]:
                for p in layer.parameters():
                    p.requires_grad = False
        if activation:
            for layer in self.linear_layers[1::2]:
                for p in layer.parameters():
                    p.requires_grad = False
        if pca_norm:
            self.pca_shift.requires_grad = False
            self.pca_scale.requires_grad = False
        if pca_basis:
            self.pca_basis.requires_grad = False
        if wavelengths:
            self.wavelengths.requires_grad = False
        if spectrum:
            self.log_spectrum_shift.requires_grad = False
            self.log_spectrum_scale.requires_grad = False

    def unfreeze(
        self,
        latents: bool = False,
        linear: bool = True,
        activation: bool = True,
        pca_norm: bool = False,
        pca_basis: bool = False,
        wavelengths: bool = False,
        spectrum: bool = False,
    ) -> None:
        """Un-freeze network parameters during training.

        Parameters
        ----------
        latents: bool, default=False
            Whether to un-freeze the parameters that normalize the latent variables.
        linear: bool, default=True
            Whether to un-freeze the weights in the linear layers
        activation: bool, default=True
            Whether to un-freeze the parameters in the activation layers
        pca_norm: bool, default=False
            Whether to un-freeze the parameters that normalize the PCA coefficients
        pca_basis: bool, default=False
            Whether to un-freeze the PCA basis vectors
        wavelengths: bool, default=False
            Whether to un-freeze the values in the wavelength grid
        spectrum: bool, default=False
            Whether to un-freeze the parameters that normalize the spectrum
        """
        if latents:
            self.latent_shift.requires_grad = True
            self.latent_scale.requires_grad = True
        if linear:
            for layer in self.linear_layers[::2]:
                for p in layer.parameters():
                    p.requires_grad = True
        if activation:
            for layer in self.linear_layers[1::2]:
                for p in layer.parameters():
                    p.requires_grad = True
        if pca_norm:
            self.pca_shift.requires_grad = True
            self.pca_scale.requires_grad = True
        if pca_basis:
            self.pca_basis.requires_grad = True
        if wavelengths:
            self.wavelengths.requires_grad = True
        if spectrum:
            self.log_spectrum_shift.requires_grad = True
            self.log_spectrum_scale.requires_grad = True


class Speculator(nn.Module):
    def __init__(
        self,
        model_file: Path | str | None = paths.models / "speculator_model.pt",
    ) -> None:
        super().__init__()

        # Load submodules
        self.uv_module = SpeculatorModule(speculator_uv_config)
        self.optical_module = SpeculatorModule(speculator_optical_config)

        # Load saved parameters
        if model_file is not None:
            self.load_state_dict(torch.load(model_file))

        # Freeze most parameters
        self.freeze()

    @property
    def wavelengths(self) -> torch.Tensor:
        """The combined wavelength tensor."""
        uv_wavelengths = self.uv_module.wavelengths
        optical_wavelengths = self.optical_module.wavelengths
        return torch.hstack((uv_wavelengths, optical_wavelengths))

    def forward(
        self,
        latents: torch.Tensor,
        log_mass: torch.Tensor,
    ) -> torch.tensor:
        """Map latents onto an SED."""
        # Construct spectra in two wavelength ranges
        uv_spectrum = self.uv_module(latents, log_mass)
        optical_spectrum = self.optical_module(latents, log_mass)

        # Combine into a single spectrum
        spectrum = torch.hstack((uv_spectrum, optical_spectrum))

        return spectrum

    def freeze(
        self,
        latents: bool = True,
        linear: bool = True,
        activation: bool = True,
        pca_norm: bool = True,
        pca_basis: bool = True,
        wavelengths: bool = True,
        spectrum: bool = True,
    ) -> None:
        """Freeze network parameters during training.

        Parameters
        ----------
        latents: bool, default=True
            Whether to freeze the parameters that normalize the latent variables.
        linear: bool, default=True
            Whether to freeze the weights in the linear layers
        activation: bool, default=True
            Whether to freeze the parameters in the activation layers
        pca_norm: bool, default=True
            Whether to freeze the parameters that normalize the PCA coefficients
        pca_basis: bool, default=True
            Whether to freeze the PCA basis vectors
        wavelengths: bool, default=True
            Whether to freeze the values in the wavelength grid
        spectrum: bool, default=True
            Whether to freeze the parameters that normalize the spectrum
        """
        self.uv_module.freeze(
            latents=latents,
            linear=linear,
            activation=activation,
            pca_norm=pca_norm,
            pca_basis=pca_basis,
            wavelengths=wavelengths,
            spectrum=spectrum,
        )
        self.optical_module.freeze(
            latents=latents,
            linear=linear,
            activation=activation,
            pca_norm=pca_norm,
            pca_basis=pca_basis,
            wavelengths=wavelengths,
            spectrum=spectrum,
        )

    def unfreeze(
        self,
        latents: bool = False,
        linear: bool = True,
        activation: bool = True,
        pca_norm: bool = False,
        pca_basis: bool = False,
        wavelengths: bool = False,
        spectrum: bool = False,
    ) -> None:
        """Un-freeze network parameters during training.

        Parameters
        ----------
        latents: bool, default=False
            Whether to un-freeze the parameters that normalize the latent variables.
        linear: bool, default=True
            Whether to un-freeze the weights in the linear layers
        activation: bool, default=True
            Whether to un-freeze the parameters in the activation layers
        pca_norm: bool, default=False
            Whether to un-freeze the parameters that normalize the PCA coefficients
        pca_basis: bool, default=False
            Whether to un-freeze the PCA basis vectors
        wavelengths: bool, default=False
            Whether to un-freeze the values in the wavelength grid
        spectrum: bool, default=False
            Whether to un-freeze the parameters that normalize the spectrum
        """
        self.uv_module.unfreeze(
            latents=latents,
            linear=linear,
            activation=activation,
            pca_norm=pca_norm,
            pca_basis=pca_basis,
            wavelengths=wavelengths,
            spectrum=spectrum,
        )
        self.optical_module.unfreeze(
            latents=latents,
            linear=linear,
            activation=activation,
            pca_norm=pca_norm,
            pca_basis=pca_basis,
            wavelengths=wavelengths,
            spectrum=spectrum,
        )


def sample_prior(N: int, seed: int) -> torch.Tensor:
    """Return samples from the latent prior distribution.

    This prior is taken from Table 2 of Alsing 2020.
    Note the order of parameters is:
    1. ln(Z/Z*)
    2-7. ln(SFR ratios), late->early
    8. sqrt(tau_2)
    9. n
    10. tau_1 / tau_2
    11. ln(f_AGN)
    12. ln(tau_AGN)
    13. ln(Z_gas/Z*)
    14. redshift
    15. stellar mass

    Parameters
    ----------
    N: int
        The number of samples to return.
    seed: int
        The random seed.

    Returns
    -------
    torch.Tensor
        The latent samples
    """
    rng = np.random.default_rng(seed)

    # First stellar mass
    log_mass = rng.uniform(7, 12.5, size=N)

    # Stellar metallicity
    # per Alsing 2020, we use the Gallazzi 2005 relation
    Zstar_pp = np.array(
        [
            [8.910e00, -6.000e-01, 5.550e-01],
            [9.110e00, -6.100e-01, 5.350e-01],
            [9.310e00, -6.500e-01, 5.250e-01],
            [9.510e00, -6.100e-01, 5.100e-01],
            [9.720e00, -5.200e-01, 5.100e-01],
            [9.910e00, -4.100e-01, 4.950e-01],
            [1.011e01, -2.300e-01, 4.700e-01],
            [1.031e01, -1.100e-01, 4.100e-01],
            [1.051e01, -1.000e-02, 3.050e-01],
            [1.072e01, 4.000e-02, 2.300e-01],
            [1.091e01, 7.000e-02, 1.900e-01],
            [1.111e01, 1.000e-01, 1.700e-01],
            [1.131e01, 1.200e-01, 1.600e-01],
            [1.151e01, 1.300e-01, 1.600e-01],
            [1.172e01, 1.400e-01, 1.600e-01],
            [1.191e01, 1.500e-01, 1.650e-01],
        ]
    )
    Zstar_mean = np.interp(log_mass, Zstar_pp[:, 0], Zstar_pp[:, 1])
    Zstar_std = np.interp(log_mass, Zstar_pp[:, 0], Zstar_pp[:, 2])
    a, b = (-1.98 - Zstar_mean) / Zstar_std, (0.19 - Zstar_mean) / Zstar_std
    Zstar = stats.truncnorm.rvs(a, b, loc=Zstar_mean, scale=Zstar_std, random_state=rng)

    # SFR ratios
    rSFR = stats.t.ppf(
        rng.uniform(stats.t.cdf(-5, df=2), stats.t.cdf(+5, df=2), size=(6, N)),
        df=2,
    )

    # Sqrt(tau_2)
    loc = 0.3
    scale = 1
    a, b = (0 - loc) / scale, (4 - loc) / scale
    tau2 = stats.truncnorm.rvs(a, b, loc=loc, scale=scale, size=N, random_state=rng)
    sqrt_tau2 = np.sqrt(tau2)

    # Calzetti index
    n = rng.uniform(-1, 0.4, size=N)

    # Tau ratio
    loc = 1
    scale = 0.3
    a, b = (0 - loc) / scale, (2 - loc) / scale
    tau_ratio = stats.truncnorm.rvs(
        a, b, loc=loc, scale=scale, size=N, random_state=rng
    )

    # Log AGN fraction
    log_fAGN = rng.uniform(-5, np.log10(3), size=N)

    # AGN optical depth
    tau_AGN = rng.uniform(np.log10(5), np.log10(150), size=N)

    # Gas metallicity
    Zgas = rng.uniform(-2, 0.5, size=N)

    # Redshift
    redshift = rng.uniform(0, 2.5, size=N)

    latents = torch.vstack(
        (
            torch.from_numpy(Zstar).float(),
            torch.from_numpy(rSFR).float(),
            torch.from_numpy(sqrt_tau2).float(),
            torch.from_numpy(n).float(),
            torch.from_numpy(tau_ratio).float(),
            torch.from_numpy(log_fAGN).float(),
            torch.from_numpy(tau_AGN).float(),
            torch.from_numpy(Zgas).float(),
            torch.from_numpy(redshift).float(),
        )
    ).T
    redshift = torch.from_numpy(np.atleast_2d(redshift)).float().T
    log_mass = torch.from_numpy(np.atleast_2d(log_mass)).float().T

    return redshift, log_mass, latents
