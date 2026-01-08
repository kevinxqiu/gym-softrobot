from typing import Optional, Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

from abc import ABC, abstractmethod

from gym_softrobot.config import RendererType
from gym_softrobot.utils.render.base_renderer import (
    BaseRenderer,
    BaseElasticaRendererSession,
)

import pkg_resources

def render_figure(fig:plt.figure):
    w, h = fig.get_size_inches()
    dpi_res = fig.get_dpi()
    w, h = int(np.ceil(w * dpi_res)), int(np.ceil(h*dpi_res))

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    data = np.asarray(canvas.buffer_rgba())[:,:,:3]
    return data

def convert_marker_size(radius, ax):
    """
    Convert marker size from radius to s (in scatter plot).

    Parameters
    ----------
    radius : np.array or float
        Array (or a number) of radius
    ax : matplotlib.Axes
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    max_axis_length = max(abs(xlim[1]-xlim[0]), abs(ylim[1]-ylim[0]))
    scaling_factor = 3.0e3 * (2*0.1) / max_axis_length
    return np.sqrt(np.pi * (scaling_factor * radius))
    #ppi = 72 # standard point size in matplotlib is 72 points per inch (ppi), no matter the dpi
    #point_whole_ax = 5 * 0.8 * ppi
    #point_radius= 2 * radius / 1.0 * point_whole_ax
    #return point_radius**2

def convert_line_width(radius, ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if hasattr(ax, "get_zlim"):
        zlim = ax.get_zlim()
        max_axis_length = max(
            abs(xlim[1] - xlim[0]),
            abs(ylim[1] - ylim[0]),
            abs(zlim[1] - zlim[0]),
        )
    else:
        max_axis_length = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
    if max_axis_length == 0:
        max_axis_length = 1.0
    scaling_factor = 150.0 / max_axis_length
    widths = radius * scaling_factor
    return np.clip(widths, 0.5, None)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class Geom(ABC):
    @abstractmethod
    def __call__(self):
        pass


class ElasticaRod(Geom):
    # RGB color must be 2d array 
    rgb_color = np.array([[0.35, 0.29, 1.0]])

    def __init__(self, rod, ax, is_2d: bool = False, plane_axes=(0, 2)):
        self.rod = rod
        self.ax = ax
        self.is_2d = is_2d
        self.plane_axes = plane_axes

        if self.is_2d:
            polygon = self.get_outline_polygon()
            edge_color = ElasticaRod.rgb_color[0] * 0.7
            self.patch = Polygon(
                polygon,
                closed=True,
                facecolor=ElasticaRod.rgb_color[0],
                edgecolor=edge_color,
                linewidth=0.6,
                joinstyle="round",
            )
            ax.add_patch(self.patch)
        else:
            # Initialize line segments
            segments, rad = self.get_segments_radius()
            self.collection = Line3DCollection(
                segments,
                colors=ElasticaRod.rgb_color[0],
                linewidths=convert_line_width(rad, ax),
            )
            ax.add_collection3d(self.collection)

    def get_segments_radius(self):
        pos = self.rod.position_collection.copy()
        rad = self.rod.radius.copy()
        pos = pos.T
        if pos.shape[0] == rad.shape[0]:
            rad = 0.5 * (rad[1:] + rad[:-1])
        segments = np.stack([pos[:-1], pos[1:]], axis=1)
        return segments, rad

    def get_outline_polygon(self):
        pos = self.rod.position_collection.copy().T
        rad = self.rod.radius.copy()
        if pos.shape[0] == rad.shape[0] + 1:
            node_rad = np.empty(pos.shape[0], dtype=rad.dtype)
            node_rad[1:-1] = 0.5 * (rad[:-1] + rad[1:])
            node_rad[0] = rad[0]
            node_rad[-1] = rad[-1]
        elif pos.shape[0] == rad.shape[0]:
            node_rad = rad
        else:
            node_rad = np.full(pos.shape[0], rad.mean(), dtype=rad.dtype)

        pos_2d = pos[:, self.plane_axes]
        seg = np.diff(pos_2d, axis=0)
        tangents = np.zeros_like(pos_2d)
        tangents[0] = seg[0]
        tangents[-1] = seg[-1]
        if seg.shape[0] > 1:
            tangents[1:-1] = seg[:-1] + seg[1:]
        norm = np.linalg.norm(tangents, axis=1, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        tangents = tangents / norm
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        left = pos_2d + normals * node_rad[:, None]
        right = pos_2d - normals * node_rad[:, None]
        return np.vstack([left, right[::-1]])

    def __call__(self):
        if self.is_2d:
            self.patch.set_xy(self.get_outline_polygon())
            return self.patch
        # Update line segments and thickness
        segments, rad = self.get_segments_radius()
        self.collection.set_segments(segments)
        self.collection.set_linewidths(convert_line_width(rad, self.ax))
        return self.collection

class ElasticaRodDirector(Geom):
    # TODO
    def __init__(self, rod, ax):
        self.rod = rod
        self.ax = ax

    def __call__(self):
        return None


class ElasticaCylinder(Geom):
    rgb_color = np.array([[0.35, 0.29, 1.0]])

    def __init__(self, body, ax, is_2d: bool = False, plane_axes=(0, 2)):
        self.body = body
        self.ax = ax
        self.is_2d = is_2d
        self.plane_axes = plane_axes

        pos1, pos2, rad = self.get_position_radius()
        end_caps = np.vstack((pos1, pos2))
        if self.is_2d:
            self.line = LineCollection(
                [end_caps[:, self.plane_axes]],
                colors=ElasticaCylinder.rgb_color[0],
                linewidths=convert_line_width(np.array([rad]), ax),
            )
            ax.add_collection(self.line)
        else:
            size = convert_marker_size(rad / 2, ax)
            self.scatter = ax.scatter(
                end_caps[:, 0],
                end_caps[:, 1],
                end_caps[:, 2],
                s=size,
                c=ElasticaCylinder.rgb_color,
            )
            #self.line, = ax.plot(end_caps[:,0], end_caps[:,1], end_caps[:,2], linewidth=size**0.5, c=ElasticaCylinder.rgb_color)

    def get_position_radius(self):
        radius = self.body.radius
        rad = radius[0] if np.ndim(radius) > 0 else radius
        length = self.body.length
        tangent = self.body.director_collection[2, :, 0]
        pos1 = self.body.position_collection[:, 0]
        pos2 = pos1 + length * tangent
        return pos1, pos2, rad

    def __call__(self):
        # Update scatter plot positions
        pos1, pos2, rad = self.get_position_radius()
        end_caps = np.vstack((pos1, pos2))
        if self.is_2d:
            self.line.set_segments([end_caps[:, self.plane_axes]])
            self.line.set_linewidths(convert_line_width(np.array([rad]), self.ax))
        else:
            self.scatter._offsets3d = end_caps[:,0], end_caps[:,1], end_caps[:,2]

        # Update line plot positions
        #self.line.set_data(end_caps[:,0], end_caps[:,1])
        #self.line.set_3d_properties(end_caps[:,2])

        # Updater radius (rigid body)
        
        #return [self.scatter, self.line]
        return [self.line] if self.is_2d else [self.scatter]


class ElasticaSphere(Geom):
    rgb_color = np.array([1.0, 0.0, 1.0])

    def __init__(self, loc, radius, ax, is_2d: bool = False, plane_axes=(0, 2)):
        if is_2d:
            self.scatter = ax.scatter(
                loc[plane_axes[0]],
                loc[plane_axes[1]],
                s=convert_marker_size(radius, ax),
                c=ElasticaSphere.rgb_color,
            )
        else:
            self.scatter = ax.scatter(
                loc[0],
                loc[1],
                loc[2],
                s=convert_marker_size(radius, ax),
                c=ElasticaSphere.rgb_color,
            )

    def __call__(self):
        return self.scatter


class Session(BaseElasticaRendererSession, BaseRenderer):
    def __init__(
        self,
        width,
        height,
        dpi=100,
        projection="3d",
        plane_axes=None,
        padding_ratio=0.05,
        axis_limits=None,
    ):
        self.object_collection = []
        self.width = width
        self.height = height
        self.dpi = dpi
        self.is_2d = projection == "2d"
        self.padding_ratio = padding_ratio
        self.axis_limits = axis_limits
        if self.is_2d:
            self.plane_axes = (0, 2) if plane_axes is None else plane_axes
        else:
            self.plane_axes = None

        px = 1.0 / dpi
        self.fig = plt.figure(
            figsize=(width*px,height*px),
            frameon=True,
            dpi=dpi,
        )
        if self.is_2d:
            self.ax = plt.axes()
            axis_labels = ["x", "y", "z"]
            self.ax.set_xlabel(axis_labels[self.plane_axes[0]])
            self.ax.set_ylabel(axis_labels[self.plane_axes[1]])
        else:
            self.ax = plt.axes(projection="3d")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")

    @property
    def type(self):
        return RendererType.MATPLOTLIB

    def add_rod(self, rod):
        self.object_collection.append(
            ElasticaRod(
                rod,
                self.ax,
                is_2d=self.is_2d,
                plane_axes=self.plane_axes,
            )
        )
        # TODO Maybe give another configuration to plot the directors
        # self.object_collection.append(ElasticaRodDirector(rod, self.ax))

    def add_rigid_body(self, body):
        self.object_collection.append(
            ElasticaCylinder(
                body,
                self.ax,
                is_2d=self.is_2d,
                plane_axes=self.plane_axes,
            )
        )

    def add_point(self, loc: list, radius: float):
        # Add static sphere
        self.object_collection.append(
            ElasticaSphere(
                loc,
                radius,
                self.ax,
                is_2d=self.is_2d,
                plane_axes=self.plane_axes,
            )
        )

    def render(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera_param: Optional[tuple] = None, # POVray parameter
        **kwargs
    ):
        # Reset width and height
        if not width:
            width = self.width
        if not height:
            height = self.height

        # Maybe convert povray cmaera_param to matplotlib viewpoint

        objects = [obj() for obj in self.object_collection]
        self.rescale_axis()
        rendered_data = render_figure(self.fig)
        return rendered_data

    def close(self):
        plt.close(plt.gcf())
        self.object_collection.clear()

    def rescale_axis(self):
        if self.axis_limits is not None:
            if self.is_2d:
                xlim, ylim = self.axis_limits
                self.ax.set_xlim(xlim[0], xlim[1])
                self.ax.set_ylim(ylim[0], ylim[1])
            else:
                xlim, ylim, zlim = self.axis_limits
                self.ax.set_xlim3d(xlim[0], xlim[1])
                self.ax.set_ylim3d(ylim[0], ylim[1])
                self.ax.set_zlim3d(zlim[0], zlim[1])
            return
        self.ax.relim()
        self.ax.autoscale_view()
        if self.is_2d:
            self.ax.set_aspect("equal", adjustable="box")
            if self.padding_ratio:
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                pad_x = (xlim[1] - xlim[0]) * self.padding_ratio
                pad_y = (ylim[1] - ylim[0]) * self.padding_ratio
                self.ax.set_xlim(xlim[0] - pad_x, xlim[1] + pad_x)
                self.ax.set_ylim(ylim[0] - pad_y, ylim[1] + pad_y)
        else:
            set_axes_equal(self.ax)
