# from __future__ import annotations # <- Uncomment if python >=3.7
import numpy as np
from abc import ABC, abstractmethod
import typing
from typing import Sequence
from functools import partial



# Adds Gaussian Noise of mean mu and variance sigma to an input value.  
class GaussianNoise:
    def __init__(self, mu=0., sigma=1.):
        self.mu = mu
        self.sigma = sigma

    def get_noise(self):
        return np.random.normal(self.mu, self.sigma**2)

    def add_noise(self, values):
        return np.random.normal(values + self.mu, self.sigma**2)

    def probability(self, x):
        return 1/(self.sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - self.mu)/self.sigma)**2)


class Particle(ABC):
    @abstractmethod
    def __add__(self, movement : tuple) -> 'Particle':
        pass

    @abstractmethod
    def get_location(self) -> tuple:
        pass

# TODO : replace list[Particle] with more generic sequence type
ParticleVector = Sequence[Particle]

class XYParticle(Particle):
    def __init__(self, location : tuple):
        self.x, self.y = location

    def __add__(self, movement : tuple) -> 'Particle':
        assert(len(movement) == 2)
        x, y = movement
        return XYParticle((
            self.x + x,
            self.y + y
        ))

    def get_location(self) -> tuple:
        return self.x, self.y

# Used to add the process noise to existing particles
class ProcessModel(ABC):
    def update(self, particles : ParticleVector, u: tuple) -> ParticleVector:
        pass


class LinearMovementModel(ProcessModel):
    def __init__(self, mu=0., sigma=1.):
        self.additive_noise = GaussianNoise(mu, sigma)

    def update(self, particles : ParticleVector, u: tuple=None) -> ParticleVector:
        n_dim = len(particles[0].get_location())
        if u is not None:
            assert(len(u) == n_dim)
        def add_noise(p : Particle):
            movement = tuple(
                    self.additive_noise.get_noise()
                    for i in range(n_dim)
            )
            if u is not None:
                p += u
            return p + movement
        return [add_noise(p) for p in particles]


# Measures p(z|x) for each x \in particles, i.e. how likelhood a measurement would be if at each particle.
class MeasurementModel(ABC):
    DataType = typing.Type 
    def likelihood(self, particles : ParticleVector, data : DataType) -> Sequence[float]:
        pass


class DirectLocationSensor(MeasurementModel):
    DataType = Particle

    def __init__(self, eps=1e-3):
        self.eps=eps

    def likelihood(self, particles : ParticleVector, data : DataType):
        y_observed = data.get_location()
        def particle_likelihood(particle):
            y_hat  = particle.get_location()
            distance = sum((
                (y_hat_i - y_observed_i)**2 
                for y_hat_i, y_observed_i in zip(y_hat, y_observed)
            ))
            return max(distance, self.eps)**-1
        return tuple(particle_likelihood(p) for p in particles)


class ParticleFilter:
    # @param n_particles Number of particles to create for filter
    # @param likelihood_func List of functions that return the likelihood of being in a location given location and input data
    # TODO: Particle range doesn't work with one specification
    # NOTE: Only (x,y) particle type is supported at the moment. 
    def __init__(self, n_particles, measurement_model, process_model, particle_factory):
        self.n_particles = n_particles
        self.measurement_model = measurement_model
        self.process_model = process_model
        self.particles = list()
        for i in range(self.n_particles):
            self.particles.append(particle_factory())
        # self.particles = np.array(self.particles)
        self.top_10 = self.particles[:10]

    def update(self, data, u=None):
        # --------------
        # Process update
        # --------------
        self.particles = self.process_model.update(self.particles)
        # --------------
        # Measurement
        # --------------
        # TODO: extend the likelihood function to be vectorized)
        likelihoods = self.measurement_model.likelihood(self.particles, data)
        total_prob = sum(likelihoods)
        likelihoods = [l / total_prob for l in likelihoods]
        # --------------
        # Resample
        # --------------
        self.top_10 = [self.particles[i] for i in np.argsort(likelihoods)[-10:][::-1].tolist()]
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=likelihoods)
        self.particles = [self.particles[i] for i in indices]

def xy_map_particle_factory(max_locations, min_locations, viable_location_map):
    viable_particle = False
    loc_y = 0
    loc_x = 0
    while not viable_particle:
        loc_x = int(np.floor(np.random.rand() * (max_locations[0] - min_locations[0]) + min_locations[0]))
        loc_y = int(np.floor(np.random.rand() * (max_locations[1] - min_locations[1]) + min_locations[1]))
        viable_particle = viable_location_map((loc_x, loc_y))
    return XYParticle((loc_x, loc_y))

if __name__ == "__main__":
    particle_range=(100., 100.)
    viable_particle_func = lambda x : True

    if np.isscalar(particle_range[0]):
        max_locations = (particle_range[0], particle_range[1])
        min_locations = (0, 0)
    else:
        max_locations = particle_range[1]
        min_locations = particle_range[0]

    particle_factory = partial(xy_map_particle_factory, max_locations, min_locations, viable_particle_func)

    pfilter = ParticleFilter(
            n_particles=20,
            measurement_model=DirectLocationSensor(),
            process_model=LinearMovementModel(mu=0., sigma=2.),
            particle_factory=particle_factory
    )

    # TODO: Realize the particle filter should actually receive a new likelihood function based on estimates
    for i in range(10):
        pfilter.update(XYParticle((50+i, 50+i)))
    for p in pfilter.particles:
        print(p.get_location())
    


