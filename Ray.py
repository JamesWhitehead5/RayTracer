import math

import numpy as np
import torch
import unittest
from numbers import Real

import matplotlib.pyplot as plt

class Ray:
    """

    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, slope: torch.Tensor, index=1., plotter=None):
        self.x = x
        self.y = y
        self.slope = slope
        self.index = index

        self.x_record = []
        self.y_record = []

        self._save_coord()

        self.plotter = plotter

    def _save_coord(self):
        """Saves the position of the ray to a record"""
        self.x_record.append(self.x.detach().numpy().copy())
        self.y_record.append(self.y.detach().numpy().copy())

    def propagate_by(self, distance: Real):
        """Propagates the ray by a fixed distance in the x direction"""
        self.y += distance * self.slope
        self.x += distance

        self._save_coord()

        return self

    def propagate_to(self, x_position: Real):
        distance = x_position - self.x
        self.propagate_by(distance)

    def refract(self, n_into: Real, interface_angle: Real):
        """
        Angles for the slope of a surface represent the angle swept from the horizontal, coutnerclockwise to the surface.
        This necessitates that the angle be in the range [0, pi)
        """
        incoming_angle = torch.arctan(self.slope)  # angle of the incident ray
        theta1 = torch.pi / 2 + incoming_angle - interface_angle
        theta2 = torch.arcsin(self.index / n_into * torch.sin(theta1))
        outgoing_angle = -torch.pi / 2 + theta2 + interface_angle

        self.slope = torch.tan(outgoing_angle)
        self.index = n_into

        return self

    @staticmethod
    def line_circle_intersection(m, b, x0, y0, r):
        """

        :param m: slope of line
        :param b: y intersect of line
        :param x0: x origin on circle
        :param y0: x origin on circle
        :param r: radius of circle
        :return:
        """

        # coefficients in the quadratic equaiton a2*x^2 + a1*x + a0 = 0
        a2 = m**2 + 1
        a1 = -2 * x0 + 2 * m * b - 2 * y0 * m
        a0 = x0**2 + b**2 - 2 * y0 * b + y0**2 - r**2

        radicand = (a1**2 - 4 * a2 * a0).type(torch.complex128)

        x1 = (-a1 - torch.sqrt(radicand)) / (2 * a2)
        x2 = (-a1 + torch.sqrt(radicand)) / (2 * a2)
        return x1, x2

    def refract_arc(self, n_into: Real, circle_x: Real, circle_y: Real, circle_r: Real):
        """

        :param n_into:
        :param circle_x:
        :param circle_y:
        :param circle_r:
        :return:
        """

        x1, x2 = Ray.line_circle_intersection(
            m=self.slope, b=self.y - self.slope * self.x, x0=circle_x, y0=circle_y, r=circle_r
        )

        # Rays that miss the surface will have their x coordinate sent to infinity.

        epsilon = 1e-11  # accounts to float error
        infinity = torch.Tensor([torch.inf]).type(torch.double)
        # complex numbers indicate the ray doesn't intersect with the circle

        x1 = torch.where(torch.abs(x1.imag) < epsilon, x1.real, infinity)
        x2 = torch.where(torch.abs(x2.imag) < epsilon, x2.real, infinity)

        # if both intersections are to the left of the ray, they will miss
        x1 = torch.where(x1 >= self.x, x1, infinity)
        x2 = torch.where(x2 >= self.x, x2, infinity)

        # only the first surface that the ray will interact with will be used
        x_intersect = torch.min(x1, x2)

        # determine the y-coordinate of the intersection
        b = self.y - self.slope * self.x
        y_intersect = self.slope * x_intersect + b

        # determine the angle of the tangent line
        interface_angle = torch.atan2(x_intersect - circle_x, y_intersect - circle_y)

        self.propagate_by(x_intersect - self.x)
        # self._save_coord()
        self.refract(n_into=n_into, interface_angle=interface_angle)

        if self.plotter is not None:
            plt.Circle((circle_x, circle_y), circle_r, fill=False)

    @staticmethod
    def angle_to_slope(degrees: torch.Tensor):
        return torch.tan(degrees / 180 * torch.pi)

    @staticmethod
    def plot_rays(ray):
        if ray.plotter is not None:
            combined_x = np.stack(ray.x_record)
            combined_y = np.stack(ray.y_record)
            ray.plotter.plot(combined_x, combined_y)
            ray.plotter.scatter(combined_x, combined_y)


class TestRay(unittest.TestCase):

    def setUp(self) -> None:
        self.ray1 = Ray(
            x=torch.tensor(0.),
            y=torch.tensor(0.),
            slope=torch.tensor(0.5),
            index=1.,
        )
        self.ray2 = Ray(
            x=torch.tensor(0.),
            y=torch.tensor(0.5),
            slope=torch.tensor(0.5),
            index=1.,
        )

    def test_prop(self):
        self.ray1.propagate_by(1.)
        self.assertAlmostEqual(self.ray1.x, 1.)
        self.assertAlmostEqual(self.ray1.y, 0.5)
        self.assertAlmostEqual(self.ray1.slope, 0.5)

    def test_circle_intersection(self):
        x1, x2 = Ray.line_circle_intersection(m=torch.tensor(0.), b=0, x0=0, y0=0, r=1.)
        self.assertAlmostEqual(x1.numpy(), -1.)
        self.assertAlmostEqual(x2.numpy(), 1.)

        x1, x2 = Ray.line_circle_intersection(m=torch.tensor(1), b=0, x0=0, y0=0, r=1.)
        self.assertAlmostEqual(x1.numpy(), -1. / math.sqrt(2.))
        self.assertAlmostEqual(x2.numpy(), 1. / math.sqrt(2.))

        x1, x2 = Ray.line_circle_intersection(m=torch.tensor(0), b=0, x0=0, y0=0.5, r=1.)
        self.assertAlmostEqual(x1.numpy(), -math.sqrt(0.75))
        self.assertAlmostEqual(x2.numpy(), math.sqrt(0.75))




