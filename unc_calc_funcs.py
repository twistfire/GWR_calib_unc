import numpy as np

def calculate_eps_air(RH, T, P, Ps):
    """
    Calculate the dielectric permittivity of air (ε_air) based on environmental conditions.

    Parameters:
    RH : float
        Relative humidity in percentage (%).
    T : float
        Absolute temperature in Kelvin (K).
    P : float
        Atmospheric pressure in mmHg.
    Ps : float
        Saturated water vapor pressure in mmHg.

    Returns:
    float
        Dielectric permittivity of air (ε_air).
    """
    # Define the permittivity of vacuum (ε_0) in F/m
    eps_0 = 8.854e-12  # Farad per meter

    if (Ps is None):
        Ps_mmHg, Ps_kPa = calculate_Ps(T_K = T)

    # Calculate the dielectric permittivity of air
    eps_air = eps_0 * (1 + 211 / T *  (P + 48 * Ps_mmHg / T * RH)*1e-6)
    
    return eps_air



def calculate_Ps(T_K):
    """
    Calculate the saturated vapor pressure (P_s) using the Buck equation with temperature in Kelvin.
    
    Parameters:
    T_K : float
        Absolute temperature in Kelvin (K).
    
    Returns:
    float
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