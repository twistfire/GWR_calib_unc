import numpy as np

def convert_P_kPa_to_P_mmHg(P_kPa: float):
    P_mmHg = P_kPa * 7.50062
    return P_mmHg 


def calculate_eps_air(RH, T, P):
    """
    Calculate the dielectric permittivity of air (ε_air) based on environmental conditions.

    Parameters:
    RH : float
        Relative humidity in percentage (%).
    T : float
        Absolute temperature in Kelvin (K).
    P : float
        Atmospheric pressure in mmHg.

    Returns:
    tuple
        Dielectric permittivity abs value (ε_abs).
        Dielectric permittivity relative to vacuum (ε_rel).
    """
    # Define the permittivity of vacuum (ε_0) in F/m
    eps_0 = 8.854e-12  # Farad per meter

    Ps_mmHg, _ = calculate_Ps(T)
    
    # Calculate the dielectric permittivity of air
    eps_abs = eps_0 * (1 + 211 / T * (P + 48 * Ps_mmHg / T * RH) * 1e-6)
    eps_rel = eps_abs / eps_0

    return eps_abs, eps_rel


def calculate_Ps(T_K):
    """
    Calculate the saturated vapor pressure (P_s) using the Buck equation with temperature in Kelvin.
    
    Parameters:
    T_K : float
        Absolute temperature in Kelvin (K).
    
    Returns:
    tuple
        Saturated vapor pressure (Ps_mmHg) in mmHg.
        Saturated vapor pressure (Ps_kPa) in kPa.
    """
    # Convert temperature from Kelvin to Celsius
    T_C = T_K - 273.15
    
    # Constants for Buck equation
    A = 17.368  # Coefficient A
    B = 238.88  # Coefficient B (in Celsius)
    
    # Calculate P_s in kPa
    Ps_kPa = 0.61121 * np.exp((A * T_C) / (B + T_C))
    
    # Convert kPa to mmHg (1 kPa = 7.50062 mmHg)
    Ps_mmHg = Ps_kPa * 7.50062
    
    return Ps_mmHg, Ps_kPa


def calculate_rel_eps_with_uncertainty(RH, T, P, delta_RH, delta_T, delta_P):
    """
    Calculate ε_air and its uncertainty using propagation of uncertainty.

    Parameters:
    RH : float
        Relative humidity (%).
    T : float
        Temperature (K).
    P : float
        Pressure (mmHg).
    delta_RH : float
        Uncertainty in RH.
    delta_T : float
        Uncertainty in T.
    delta_P : float
        Uncertainty in P.

    Returns:
    tuple
        (eps_air, delta_eps_air)
    """
    eps_0 = 8.854e-12  # F/m
    Ps_mmHg, _ = calculate_Ps(T)
    
    # Calculate ε_air
    eps_air = eps_0 * (1 + 211 / T * (P + 48 * Ps_mmHg / T * RH) * 1e-6)
    
    # Partial derivatives
    d_Ps_dT = calculate_dPs_dT(T)  # Derivative of Ps with respect to T
    
    d_eps_dT = (
        -eps_0 * (211 / T**2) * (P + 48 * Ps_mmHg / T * RH) * 1e-6
        - eps_0 * (211 / T**3) * 48 * Ps_mmHg * RH * 1e-6
        + eps_0 * (211 / T**2) * 48 * d_Ps_dT * RH * 1e-6
    )
    d_eps_dRH = eps_0 * (211 / T**2) * 48 * Ps_mmHg * 1e-6
    d_eps_dP = eps_0 * (211 / T) * 1e-6

    # Propagate uncertainty
    delta_eps_air = np.sqrt(
        (d_eps_dT * delta_T)**2 +
        (d_eps_dRH * delta_RH)**2 +
        (d_eps_dP * delta_P)**2
    )

    eps_rel = eps_air / eps_0
    delta_eps_rel = delta_eps_air / eps_0

    return eps_rel, delta_eps_rel


def calculate_dPs_dT(T_K):
    """
    Calculate the derivative of Ps with respect to T using the Buck equation.

    Parameters:
    T_K : float
        Absolute temperature in Kelvin (K).
    
    Returns:
    float
        Derivative of Ps with respect to T (in mmHg/K).
    """
    T_C = T_K - 273.15
    A = 17.368
    B = 238.88

    # Calculate derivative
    dPs_dT_kPa = 0.61121 * np.exp((A * T_C) / (B + T_C)) * (
        (A * B) / ((B + T_C)**2)
    )
    dPs_dT_mmHg = dPs_dT_kPa * 7.50062  # Convert from kPa to mmHg

    return dPs_dT_mmHg