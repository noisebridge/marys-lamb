import numpy as np

# Adds Gaussian Noise of mean mu and variance sigma to an input value.
class GaussianNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def add_noise(self, values):
        return np.random.normal(values + self.mu, self.sigma**2)

    def probability(self, x):
        return 1/(self.sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - self.mu)/self.sigma)**2)

# TODO: Movement has not been defined yet
class ParticleFilter:
    # @param n_particles Number of particles to create for filter
    # @param likelihood_func List of functions that return the likelihood of being in a location given location and input data
    # TODO: Particle range doesn't work with one specification
    # NOTE: Only Roomba coordinates (3 dof) are supported at the moment. 
    def __init__(self, n_particles, particle_range, likelihood_func, viable_particle_func = None):
        self.n_particles = n_particles
        self.likelihood = likelihood_func
        if viable_particle_func is None:
            self.viable_particle_func = lambda x : True
        else:
            self.viable_particle_func = viable_particle_func
        if np.isscalar(particle_range[0]):
            self.max_locations = (particle_range[0], particle_range[1])
            self.min_locations = (0, 0)
        else:
            self.max_locations = particle_range[1]
            self.min_locations = particle_range[0]
        self.particles = list()
        for i in range(self.n_particles):
            self.particles.append(self._init_particle(self.max_locations, self.min_locations))
        self.particles = np.array(self.particles)
        self.top_10 = self.particles[:10]
        self.move_noise = GaussianNoise(0, 2)

    def _init_particle(self, max_locations, min_locations=(0,0)):
        viable_particle = False
        loc_y = 0
        loc_x = 0
        while not viable_particle:
            loc_x = int(np.floor(np.random.rand() * (max_locations[0] - min_locations[0]) + min_locations[0]))
            loc_y = int(np.floor(np.random.rand() * (max_locations[1] - min_locations[1]) + min_locations[1]))
            # Add heading
            heading = 0.
            viable_particle = self.viable_particle_func((loc_x, loc_y, heading))
        return (loc_x, loc_y, heading)

    def update(self, data):
        # TODO: extend the likelihood function to be vectorized)
        likelihoods = list()
        for particle in self.particles:
            likelihoods.append(self.likelihood(particle, data))
        total_prob = sum(likelihoods)
        likelihoods = [l / total_prob for l in likelihoods]
        self.top_10 = self.particles[np.argsort(likelihoods)[-10:][::-1]]
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=likelihoods)
        self.particles = self.particles[indices]


if __name__ == "__main__":
    def likelihood(y_hat, y_observed):
        eps=1e-3
        distance = sum(((y_hat_i - y_observed_i)**2 for y_hat_i, y_observed_i in zip(y_hat, y_observed)))
        return max(distance, eps)**-1

    pfilter = ParticleFilter(
            n_particles=20,
            particle_range=(100, 100),
            likelihood_func=likelihood
    )

    # TODO: Realize the particle filter should actually receive a new likelihood function based on estimates
    for i in range(10):
        pfilter.update((50, 50))
    print(pfilter.particles)
    


