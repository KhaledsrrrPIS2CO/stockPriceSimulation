"""Step 1: What is Geometric Brownian Motion?
This is an abstract concept so I want to explain what GBM is all about.

Brownian motion comes from physics. It describes the random movement of particles in a substance. A Wiener process is a one-dimentional Brownian motion. It's named after Norbert Wiener who won a Nobel Prize studying one-dimentional Brownian motions.

The Wiener process features prominently in quantitative finance because of some useful mathemetical properties.

The GBM is a continuous-time stochastic process  where the log of the random variable follows the Wiener process with drift.

What?

It’s a data series that trends up or down through time with a defined level of volatility.

And it’s perfect for simulating stock prices.
"""

# Step 2: Import the libraries

import numpy as np
import matplotlib.pyplot as plt

# Step 3: Set the input parameters
# To simulate stock prices, we need some input parameters.

# setup params for brownian motion
s0 = 131.00
sigma = 0.25
mu = 0.35

# setup the simulation
paths = 1000
delta = 1.0 / 252.0
time = 252 * 5


# Step 4: Build the functions

def wiener_process(delta, sigma, time, paths):
    """Returns a Wiener process

    Parameters
    ----------
    delta : float
        The increment to downsample sigma
    sigma : float
        Percentage volatility
    time : int
        Number of samples to create
    paths : int
        Number of price simulations to create

    Returns
    -------
    wiener_process : np.ndarray

    Notes
    -----
    This method returns a Wiener process.
    The Wiener process is also called Brownian
    motion. For more information about the
    Wiener process check out the Wikipedia
    page: http://en.wikipedia.org/wiki/Wiener_process
    """

    # return an array of samples from a normal distribution
    return sigma * np.random.normal(loc=0, scale=np.sqrt(delta), size=(time, paths))


# Next, I define a function that creates the GBM returns.
def gbm_returns(delta, sigma, time, mu, paths):
    """Returns from a Geometric brownian motion

    Parameters
    ----------
    delta : float
        The increment to downsample sigma
    sigma : float
        Percentage volatility
    time : int
        Number of samples to create
    mu : float
        Percentage drift
    paths : int
        Number of price simulations to create

    Returns
    -------
    gbm_returns : np.ndarray

    Notes
    -----
    This method constructs random Geometric Brownian
    Motion (GBM).
    """
    process = wiener_process(delta, sigma, time, paths)
    return np.exp(
        process + (mu - sigma ** 2 / 2) * delta
    )


# Finally, I prepend a row of 1s to the returns array and multiply the starting stock
# price by the cumulative product of the GBM returns to produce the price paths.

def gbm_levels(s0, delta, sigma, time, mu, paths):
    """Returns price paths starting at s0

    Parameters
    ----------
    s0 : float
        The starting stock price
    delta : float
        The increment to downsample sigma
    sigma : float
        Percentage volatility
    time : int
        Number of samples to create
    mu : float
        Percentage drift
    paths : int
        Number of price simulations to create

    Returns
    -------
    gbm_levels : np.ndarray
    """
    returns = gbm_returns(delta, sigma, time, mu, paths)

    stacked = np.vstack([np.ones(paths), returns])
    return s0 * stacked.cumprod(axis=0)


# Step 4: Visualize the results

price_paths = gbm_levels(s0, delta, sigma, time, mu, paths)
plt.plot(price_paths, linewidth=0.25)
plt.show()
