import torch
import numpy as np
import matplotlib.pyplot as plt

from Ray import Ray

class Model:
    def __init__(self, plotter):
        n_rays = 11  # number of rays to trace
        source_aperture = 5.
        self.d1 = 10
        self.d2 = 20
        self.n_glass = 1.51
        self.nominal_focal_length = 10.
        self.lens_center_thickness = 1.


        # starting x and y positions of the rays
        x = torch.zeros(size=(n_rays,))
        y = torch.linspace(-source_aperture / 2., source_aperture / 2., n_rays)
        angle = torch.zeros(size=(n_rays,))  # initial slopes of the rays.

        # initialize the rays
        rays = Ray(x=x, y=y, angle=angle, plotter=plotter)
        rays = rays.propagate_to(self.d1)
        rays = rays.refract(n_into=self.n_glass, interface_normal_angle=0.)  # refract through a vertical wall of glass
        self.rays = rays

    def forward(self, lens_radius):
        rays = self.rays.refract_arc(
            n_into=1.,
            circle_x=self.d1 - lens_radius + self.lens_center_thickness,
            circle_y=0.,
            circle_r=lens_radius,
        )
        rays = rays.propagate_to(self.d2)
        return rays

def loss(rays):
    """"""

    # takes the distribution of intersections of the rays with the final plane and multiplies it with a gaussian function
    y = rays.y[torch.abs(rays.y) < torch.inf]

    mu = 0.  # location of maximum
    sigma = 1. # related to width of the gaussian
    gaussian = torch.exp(-1./2 * (y - mu)**2/sigma**2)

    return -torch.sum(gaussian)

if __name__ == '__main__':
    rt_model = Model(plotter=plt)
    initial_lens_radius = rt_model.nominal_focal_length * (rt_model.n_glass - 1.)  # calculated using lensmaker's equaiton
    lens_radius = torch.tensor(initial_lens_radius, requires_grad=True)
    rt_model.forward(lens_radius)

    rt_model.rays.plotter = None  # disable plotting for optimization

    learning_rates = [0.1, 0.01]

    for learning_rate in learning_rates:
        for _ in range(10):

            rays = rt_model.forward(lens_radius)
            current_loss = loss(rays)
            current_loss.backward()

            with torch.no_grad():
                lens_radius -= lens_radius.grad * learning_rate
            lens_radius.grad = None
            print(current_loss)



    # plot result
    plt.figure()
    model = Model(plotter=plt)
    model.forward(lens_radius)
    plt.show()
