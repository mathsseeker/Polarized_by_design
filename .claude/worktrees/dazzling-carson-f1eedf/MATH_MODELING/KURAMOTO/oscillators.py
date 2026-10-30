import numpy as np
def initialize_oscillators(num_oscillators: int, sigma: float = 1.0):
    

    # Assign a random initial phase to each oscillator
    theta = np.random.uniform(0, 2 * np.pi, num_oscillators)

    # Assign a random natural frequency to each oscillator
    omega = np.random.normal(0, sigma, num_oscillators)
    return theta, omega

# Initialize oscillators at module level so they can be imported
num_oscillators = 100
theta, omega = initialize_oscillators(num_oscillators)
