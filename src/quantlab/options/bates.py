from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantlab.options.heston import HestonParams, carr_madan_call_price, heston_characteristic_function

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BatesParams:
    """Heston stochastic volatility plus Merton-style lognormal jumps."""

    heston: HestonParams
    jump_intensity: float
    jump_mean: float
    jump_volatility: float

    def validate(self) -> None:
        self.heston.validate()
        if self.jump_intensity < 0:
            raise ValueError("jump_intensity must be non-negative")
        if self.jump_volatility < 0:
            raise ValueError("jump_volatility must be non-negative")


def bates_characteristic_function(
    u: complex,
    params: BatesParams,
    spot: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
) -> complex:
    """Characteristic function for Bates jump-diffusion stochastic volatility."""
    params.validate()
    heston_cf = heston_characteristic_function(u, params.heston, spot, maturity, rate, dividend)
    jump_compensator = np.exp(params.jump_mean + 0.5 * params.jump_volatility**2) - 1.0
    jump_cf = np.exp(
        params.jump_intensity
        * maturity
        * (
            np.exp(1j * u * params.jump_mean - 0.5 * params.jump_volatility**2 * u**2)
            - 1.0
            - 1j * u * jump_compensator
        )
    )
    return complex(heston_cf * jump_cf)


def bates_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    params: BatesParams,
    dividend: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Price a European option under the Bates model."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    cf = lambda u: bates_characteristic_function(u, params, spot, maturity, rate, dividend)
    call = carr_madan_call_price(cf, strike, maturity, rate)
    if option_type == "call":
        return call
    return float(call - spot * np.exp(-dividend * maturity) + strike * np.exp(-rate * maturity))
