import numpy as np
from typing import Tuple

def bootstrap_stats(data: np.ndarray, n_resamples: int = 1000) -> Tuple[float, float]:
    """
    Computes the mean and the bootstrap estimate of the standard deviation of the mean.
    
    This is useful for estimating the precision of a success rate or mean reward,
    especially for binary (0/1) outcomes where the sample standard deviation is naturally high.
    
    Args:
        data: A NumPy array containing the experiment results.
        n_resamples: The number of bootstrap resamples to perform.
        
    Returns:
        A tuple of (mean, bootstrap_std_of_mean).
    """
    if data.size == 0:
        return 0.0, 0.0
    
    n = len(data)
    # The actual mean of our data
    mean = np.mean(data)
    
    if n <= 1:
        return float(mean), 0.0
    
    # Perform bootstrapping to estimate the standard deviation of the mean (SEM)
    # 1. Create a matrix of random indices for resampling with replacement
    resample_idxs = np.random.randint(0, n, (n_resamples, n))
    
    # 2. Resample the data and calculate the mean for each resample set
    resampled_means = np.mean(data[resample_idxs], axis=1)
    
    # 3. The standard deviation of these resampled means is an estimate of the SEM
    bootstrap_std = np.std(resampled_means)
    
    return float(mean), float(bootstrap_std)
