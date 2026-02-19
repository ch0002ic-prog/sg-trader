def project_initial_investment(
    initial: float, years: int, cagr_min: float, cagr_max: float
) -> tuple[float, float]:
    low = initial * (1 + cagr_min) ** years
    high = initial * (1 + cagr_max) ** years
    return low, high
