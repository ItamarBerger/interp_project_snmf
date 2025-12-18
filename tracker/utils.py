
def get_run_name(cfg: dict) -> str:
    """
    Generate a name for the final run based on the configuration.

    Parameters:
    - cfg: The configuration dictionary.

    Returns:
    - A string representing the name of the final run.
    """
   # Currently returns only one thing, but we might change this based on experiment, so better to leave this as a function.
    name_parts = [
        cfg["model_name"].replace("/", "-"),
        cfg['sparsity'],
        f"ranks-{cfg['ranks'].replace(',', '-')}",
        cfg["factorization_mode"],
        f"layers-{cfg['layers'].replace(',', '-')}",
    ]

    return "_".join(map(str, name_parts))