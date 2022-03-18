import torch

from Ray import Ray
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure()

    n_rays = 11  # number of rays to trace

    source_aperture = 5.

    # starting x and y positions of the rays
    x = torch.zeros(size=(n_rays,))
    y = torch.linspace(-source_aperture/2., source_aperture/2., n_rays)
    angle = torch.zeros(size=(n_rays,))  # initial slopes of the rays.

    # initialize the rays
    rays = Ray(x=x, y=y, angle=angle, plotter=plt)

    d1 = 10
    d2 = 30
    n_glass = 1.51
    nominal_focal_length = 10.
    lens_center_thickness = 1.

    lens_radius = nominal_focal_length * (n_glass - 1.)

    rays = rays.propagate_to(d1)  # propagate by 10cm
    rays = rays.refract(n_into=n_glass, interface_normal_angle=0.)  # refract through a vertical wall of glass
    rays = rays.refract_arc(
        n_into=1.,
        circle_x=d1 - lens_radius + lens_center_thickness,
        circle_y=0.,
        circle_r=lens_radius,
    )

    rays = rays.propagate_to(d2)

    plt.gca().set_aspect('equal')
    plt.show()

    # plt.figure()
    # plt.hist(rays.y.numpy(), bins=30)
    # plt.show()