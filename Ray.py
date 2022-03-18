import copy
import math

import numpy as np
import torch
import unittest
from numbers import Real

import matplotlib.pyplot as plt

class Ray:
    """
        Represents a collection of rays in 2D. Rays propagate from left to right. They can interact with dielectric
        surfaces at an angle. All of the ray is transmitted upon refraction.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, angle: torch.Tensor, plotter=None, index=1.):
        """
        :param x: x coordinates of a point on each ray
        :param y: y coordinates of a point on each ray
        :param slope: slope of each ray
        :param index: refractive index of the material that the rays is currently in
        """
        self.x = x  # list of x coordinates
        self.y = y  # list of y coordinates
        self.angle = angle
        self.index = index

        self.plotter = plotter

        # self._save_coord()

    # def _save_coord(self):
    #     """Saves the position of the ray to a record for later reference"""
    #     with torch.no_grad():
    #         self.x_record.append(self.x.detach().numpy().copy())
    #         self.y_record.append(self.y.detach().numpy().copy())

    def propagate_by(self, distance: Real):
        """
        Propagates the ray by a fixed distance in the x direction
        :param distance: distance to propagate
        :return:
        """

        mod_ray = copy.copy(self)
        mod_ray.y = self.y + distance * torch.tan(self.angle)
        mod_ray.x = self.x + distance

        if self.plotter is not None:
            combined_x = np.stack([self.x.detach().numpy(), mod_ray.x.detach().numpy()])
            combined_y = np.stack([self.y.detach().numpy(), mod_ray.y.detach().numpy()])
            self.plotter.plot(combined_x, combined_y)
            self.plotter.scatter(mod_ray.x.detach().numpy(), mod_ray.y.detach().numpy())

        return mod_ray

    def propagate_to(self, x_position: Real):
        """
        Propagates the rays to an x position
        :param x_position: position to propagate to
        :return:
        """

        distance = x_position - self.x
        return self.propagate_by(distance)

    def refract(self, n_into: Real, interface_normal_angle: Real):
        """
        :param n_into: refractive index of the material the rays are transferring into
        :param interface_angle: the angle that the interface makes with the horizontal in radians.
        Angles for the slope of a surface represent the angle swept from the horizontal, coutnerclockwise to the surface
        :return:
        """
        theta1 = self.angle - interface_normal_angle
        # mag_theta1 = torch.abs(theta1)
        theta2 = torch.arcsin(self.index / n_into * torch.sin(theta1))

        mod_ray = copy.copy(self)
        mod_ray.angle = theta2 + interface_normal_angle
        mod_ray.index = n_into

        return mod_ray

    @staticmethod
    def line_circle_intersection(m, b, x0, y0, r):
        """
        Calculate the first intersection that a line makes with a circle in the positive x direction

        :param m: slope of line
        :param b: y intersect of line
        :param x0: x origin on circle
        :param y0: x origin on circle
        :param r: radius of circle
        :return:
        """

        # coefficients in the quadratic equation a2*x^2 + a1*x + a0 = 0
        a2 = m**2 + 1
        a1 = -2 * x0 + 2 * m * b - 2 * y0 * m
        a0 = x0**2 + b**2 - 2 * y0 * b + y0**2 - r**2

        radicand = (a1**2 - 4 * a2 * a0).type(torch.complex128)

        x1 = (-a1 - torch.sqrt(radicand)) / (2 * a2)
        x2 = (-a1 + torch.sqrt(radicand)) / (2 * a2)
        return x1, x2

    def refract_arc(self, n_into: Real, circle_x: Real, circle_y: Real, circle_r: Real):
        """
        Refracts the rays with the first surface of the arc that it interacts with

        :param n_into: refractive index of the material after the interface
        :param circle_x: x coordinate of the center of the circle
        :param circle_y: y coordinate of the center of the circle
        :param circle_r: radius of the circle
        :return:
        """

        slope = torch.tan(self.angle)
        x1, x2 = Ray.line_circle_intersection(
            m=slope, b=self.y - slope * self.x, x0=circle_x, y0=circle_y, r=circle_r
        )

        # Rays that miss the surface will have their x coordinate set to infinity.

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
        b = self.y - slope * self.x
        y_intersect = slope * x_intersect + b

        # determine the angle of the tangent line
        interface_normal_angle = torch.arctan((y_intersect - circle_y) / (x_intersect - circle_x))

        if self.plotter is not None:
            dx = torch.cos(interface_normal_angle)
            dy = torch.sin(interface_normal_angle)

            for i in range(len(x_intersect)):
                self.plotter.arrow(
                    x=x_intersect.detach().numpy()[i],
                    y=y_intersect.detach().numpy()[i],
                    dx=dx.detach().numpy()[i],
                    dy=dy.detach().numpy()[i],
                )

        mod_ray = self.propagate_by(x_intersect - self.x)

        return mod_ray.refract(n_into=n_into, interface_normal_angle=interface_normal_angle)

    @staticmethod
    def angle_to_slope(degrees: torch.Tensor):
        return torch.tan(degrees / 180 * torch.pi)

    # @staticmethod
    # def plot_rays(ray, plotter):
    #     combined_x = np.stack(ray.x_record)
    #     combined_y = np.stack(ray.y_record)
    #     plotter.plot(combined_x, combined_y)
    #     plotter.scatter(combined_x, combined_y)


class TestRay(unittest.TestCase):

    def setUp(self) -> None:
        self.ray1 = Ray(
            x=torch.tensor(0.),
            y=torch.tensor(0.),
            angle=torch.arctan(torch.tensor(0.5)),
            index=1.,
        )
        self.ray2 = Ray(
            x=torch.tensor(0.),
            y=torch.tensor(0.5),
            angle=torch.arctan(torch.tensor(0.5)),
            index=1.,
        )

    def test_prop(self):
        ray1 = self.ray1.propagate_by(1.)
        self.assertAlmostEqual(ray1.x.numpy(), 1.)
        self.assertAlmostEqual(ray1.y.numpy(), 0.5)
        self.assertAlmostEqual(ray1.angle, torch.arctan(torch.tensor(0.5)))

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
    #
    # def test_high_to_low_ref1(self):
    #
    #     rays = Ray(
    #         x=torch.tensor(1.),
    #         y=torch.tensor(0.),
    #         angle=torch.arctan(torch.tensor(-0.1)),
    #         index=2.,
    #     )
    #     rays = rays.propagate_by(1.)
    #     rays = rays.refract(1., torch.pi/2.)
    #
    #     theta1 = torch.arctan(torch.tensor(-0.1))
    #     theta2 = torch.arcsin(2. * torch.sin(theta1))
    #     slope = torch.tan(theta2)
    #
    #
    #     self.assertAlmostEqual(rays.slope.numpy(), slope.numpy(), places=6)
    #
    # def test_high_to_low_ref2(self):
    #     rays = Ray(
    #         x=torch.tensor(1.),
    #         y=torch.tensor(0.),
    #         angle=torch.arctan(torch.tensor(-0.1)),
    #         index=2.,
    #     )
    #     rays = rays.propagate_by(1.)
    #     rays = rays.refract(1., torch.pi / 2.)
    #
    #     theta1 = torch.arctan(torch.tensor(-0.1))
    #     theta2 = torch.arcsin(2. * torch.sin(theta1))
    #     slope = torch.tan(theta2)
    #
    #     self.assertAlmostEqual(rays.slope.numpy(), slope.numpy(), places=6)
    #









