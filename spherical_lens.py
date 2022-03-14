import torch
from scipy.stats import gaussian_kde

from Ray import Ray
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure()

    #  manual tests
    n_rays = 100
    x = torch.zeros(size=(n_rays,))
    y = torch.zeros(size=(n_rays,))
    slope = torch.tan(torch.linspace(-20, 20, n_rays) * torch.pi / 180)
    rays = Ray(x=x, y=y, slope=slope, plotter=plt)

    rays.refract_arc(n_into=1.57, circle_x=1.57, circle_y=0., circle_r=2.57)
    rays.propagate_to(4.5)
    rays.refract(n_into=1., interface_angle=torch.pi/2)
    rays.propagate_to(10.)

    plt.figure()
    Ray.plot_rays(rays)
    plt.show()

    plt.figure()
    kde = gaussian_kde(rays.y)
    plt.plot(kde)
    plt.show()