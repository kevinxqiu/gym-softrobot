__doc__ = """
Module contains elastica interface to create octopus model.

We tried to select the mechanical parameters within the plausible range in soft-matter studies
and simplified model. We encourage users to try out different parameters for properties, but
the behavior of physical simulator (PyElastica) might be very different.

"""

from typing import Optional

import numpy as np

from elastica import *

from gym_softrobot.utils.custom_elastica.joint import FixedJoint2Rigid
from gym_softrobot.utils.custom_elastica.constraint import BodyBoundaryCondition
from gym_softrobot.utils.actuation.actuations.muscles.longitudinal_muscle import (
    LongitudinalMuscle,
)
from gym_softrobot.utils.actuation.actuations.muscles.transverse_muscle import (
    TransverseMuscle,
)

from scipy.spatial.transform import Rotation as Rot

_OCTOPUS_PROPERTIES = {  # default parameters
    # Arm properties
    "youngs_modulus": 1e6,
    "density": 1000.0,
    # "nu": 1e-2,  # Deprecated
    # "poisson_ratio": 0.5,  # Deprecated
    # Head properties
    "body_arm_k": 1e6,
    "body_arm_kt": 1e0,
    "head_radius": 0.04,
    "head_density": 700.0,
    # Friction Properties
    "friction_multiplier": 1.00,
    "friction_symmetry": False,
}

_DEFAULT_SCALE_LENGTH = {
    "base_length": 0.2,
    "base_radius": 0.35 * 0.02,
}


def build_octopus(
    simulator,
    n_arm: int = 8,
    n_elem: int = 11,
    time_step: float = 7e-5,
    override_params: Optional[dict] = None,
):
    """Import default parameters (overridable)"""
    param = _OCTOPUS_PROPERTIES.copy()  # Always copy parameter for safety
    if isinstance(override_params, dict):
        param.update(override_params)
    """ Import default parameters (non-overridable) """
    arm_scale_param = _DEFAULT_SCALE_LENGTH.copy()

    """ Set up an arm """
    L0 = arm_scale_param["base_length"]
    r0 = arm_scale_param["base_radius"]

    rigid_rod_length = r0 * 2
    rigid_rod_radius = param["head_radius"]

    rotation_angle = 360 / n_arm
    angle_list = [rotation_angle * arm_i for arm_i in range(n_arm)]

    shearable_rods = []  # arms
    for arm_i in range(n_arm):
        arm_angle = angle_list[arm_i]
        rot = Rot.from_euler("z", arm_angle, degrees=True)
        arm_pos = rot.apply([rigid_rod_radius, 0.0, 0.0])
        arm_dir = rot.apply([1.0, 0.0, 0.0])
        rod = CosseratRod.straight_rod(
            n_elements=n_elem,
            start=arm_pos,
            direction=arm_dir,
            normal=np.array([0.0, 0.0, 1.0]),
            **arm_scale_param,
            **param,
            # nu_for_torques=damp_coefficient*((radius_mean/radius_base)**4),
        )
        shearable_rods.append(rod)
        simulator.append(rod)

    """ Add head """
    start = np.zeros((3,))
    start[2] = -r0
    direction = np.array([0.0, 0.0, 1.0])
    normal = np.array([0.0, 1.0, 0.0])
    binormal = np.cross(direction, normal)
    base_area = np.pi * rigid_rod_radius**2
    density = param["head_density"]

    rigid_rod = Cylinder(
        start, direction, normal, rigid_rod_length, rigid_rod_radius, density
    )
    simulator.append(rigid_rod)

    """ Constraint body """
    simulator.constrain(rigid_rod).using(
        # Upright rigid rod need restoration force/torque against the floor
        BodyBoundaryCondition,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    """ Set up boundary conditions """
    for arm_i in range(n_arm):
        _k = param["body_arm_k"]
        _kt = param["body_arm_kt"]
        simulator.connect(
            first_rod=rigid_rod,
            second_rod=shearable_rods[arm_i],
            first_connect_idx=-1,
            second_connect_idx=0,
        ).using(
            FixedJoint2Rigid,
            k=_k,
            nu=1e-3,
            kt=_kt,
            angle=angle_list[arm_i],
            radius=rigid_rod_radius,
        )

    """Add gravity forces"""
    _g = -9.81
    gravitational_acc = np.array([0.0, 0.0, _g])
    for arm_i in range(n_arm):
        simulator.add_forcing_to(shearable_rods[arm_i]).using(
            GravityForces, acc_gravity=gravitational_acc
        )
    # simulator.add_forcing_to(rigid_rod).using(
    #             GravityForces, acc_gravity=gravitational_acc
    #         )

    """ Add damping """
    damping_constant = 1e-2
    for arm_i in range(n_arm):
        simulator.dampen(arm_i).using(
            AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=time_step,
        )

    """ Add drag force """
    # dl = L0 / n_elem
    # fluid_factor = 1
    # r_bar = (radius_base + radius_tip) / 2
    # sea_water_dentsity = 1022
    # c_per = 0.41 / sea_water_dentsity / r_bar / dl * fluid_factor
    # c_tan = 0.033 / sea_water_dentsity / np.pi / r_bar / dl * fluid_factor
    #
    # simulator.add_forcing_to(self.shearable_rod).using(
    #     DragForce,
    #     rho_environment=sea_water_dentsity,
    #     c_per=c_per,
    #     c_tan=c_tan,
    #     system=self.shearable_rod,
    #     step_skip=self.step_skip,
    #     callback_params=self.rod_parameters_dict
    # )

    """Add friction forces (always the last thing before finalize)"""
    normal = np.array([0.0, 0.0, 1.0])
    period = 2.0

    origin_plane = np.array([0.0, 0.0, -r0])
    normal_plane = normal
    slip_velocity_tol = 1e-8
    froude = 0.1
    mu = L0 / (period * period * np.abs(_g) * froude)
    if param["friction_symmetry"]:
        kinetic_mu_array = (
            np.array([mu, mu, mu]) * param["friction_multiplier"]
        )  # [forward, backward, sideways]
    else:
        kinetic_mu_array = (
            np.array([mu, 1.5 * mu, 2.0 * mu]) * param["friction_multiplier"]
        )  # [forward, backward, sideways]
    static_mu_array = 2 * kinetic_mu_array
    for arm_i in range(n_arm):
        simulator.add_forcing_to(shearable_rods[arm_i]).using(
            AnisotropicFrictionalPlane,
            k=1e2,
            nu=1e1,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
    """
    mu = L0 / (period * period * np.abs(_g) * froude)
    kinetic_mu_array = np.array([mu, mu, mu])  # [forward, backward, sideways]
    static_mu_array = 2 * kinetic_mu_array
    simulator.add_forcing_to(rigid_rod).using(
        AnisotropicFrictionalPlaneRigidBody,
        k=8e2,
        nu=1e1,
        plane_origin=origin_plane,
        plane_normal=normal_plane,
        slip_velocity_tol=slip_velocity_tol,
        static_mu_array=static_mu_array,
        kinetic_mu_array=kinetic_mu_array,
    )
    """

    return shearable_rods, rigid_rod


def build_arm(
    simulator,
    n_elem: int = 11,
    time_step: float = 7e-5,
    override_params: Optional[dict] = None,
    attach_head: bool = None,  # TODO: To be implemented
    attach_weight: Optional[bool] = None,  # TODO: To be implemented
    start: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    normal: Optional[np.ndarray] = None,
    add_ground: bool = True,
    taper_ratio: Optional[float] = None,
    base_length: Optional[float] = None,
    damping_constant: Optional[float] = None,
    fluid_density: Optional[float] = None,
    drag_coeff_per: Optional[float] = None,
    drag_coeff_tan: Optional[float] = None,
    drag_step_skip: int = 1,
    drag_callback_params: Optional[dict] = None,
):
    """Import default parameters (overridable)"""
    param = _OCTOPUS_PROPERTIES.copy()  # Always copy parameter for safety
    if isinstance(override_params, dict):
        param.update(override_params)
    """ Import default parameters (non-overridable) """
    arm_scale_param = _DEFAULT_SCALE_LENGTH.copy()
    if base_length is not None:
        if base_length <= 0.0:
            raise ValueError("base_length must be positive")
        arm_scale_param["base_length"] = base_length

    """ Set up an arm """
    L0 = arm_scale_param["base_length"]
    r0 = arm_scale_param["base_radius"]

    arm_pos = np.array([0.0, 0.0, 0.0]) if start is None else start
    arm_dir = np.array([1.0, 0.0, 0.0]) if direction is None else direction
    rod_normal = np.array([0.0, 0.0, 1.0]) if normal is None else normal
    radius_profile = arm_scale_param["base_radius"]
    if taper_ratio is not None:
        if taper_ratio <= 0.0:
            raise ValueError("taper_ratio must be positive")
        radius_profile = np.linspace(
            arm_scale_param["base_radius"],
            arm_scale_param["base_radius"] * taper_ratio,
            n_elem,
        )
    if np.isscalar(radius_profile):
        radius_base = float(radius_profile)
        radius_tip = float(radius_profile)
    else:
        radius_base = float(radius_profile[0])
        radius_tip = float(radius_profile[-1])
    rod = CosseratRod.straight_rod(
        n_elements=n_elem,
        start=arm_pos,
        direction=arm_dir,
        normal=rod_normal,
        base_length=arm_scale_param["base_length"],
        base_radius=radius_profile,
        **param,
    )
    simulator.append(rod)

    """Add gravity forces"""
    _g = -9.81
    gravitational_acc = np.array([0.0, 0.0, _g])
    simulator.add_forcing_to(rod).using(GravityForces, acc_gravity=gravitational_acc)

    """Add friction forces (always the last thing before finalize)"""
    contact_k = 1e2  # TODO: These need to be global parameter to tune
    contact_nu = 1e1
    period = 2.0
    origin_plane = np.array([0.0, 0.0, -r0])
    slip_velocity_tol = 1e-8
    froude = 0.1
    mu = L0 / (period * period * np.abs(_g) * froude)
    if param["friction_symmetry"]:
        kinetic_mu_array = (
            np.array([mu, mu, mu]) * param["friction_multiplier"]
        )  # [forward, backward, sideways]
    else:
        kinetic_mu_array = (
            np.array([mu, 1.5 * mu, 2.0 * mu]) * param["friction_multiplier"]
        )  # [forward, backward, sideways]
    static_mu_array = 2 * kinetic_mu_array
    if add_ground:
        plane_normal = np.array([0.0, 0.0, 1.0])
        simulator.add_forcing_to(rod).using(
            AnisotropicFrictionalPlane,
            k=contact_k,
            nu=contact_nu,
            plane_origin=origin_plane,
            plane_normal=plane_normal,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )

    if damping_constant is None:
        damping_constant = 1e-2
    if damping_constant < 0.0:
        raise ValueError("damping_constant must be non-negative")
    simulator.dampen(rod).using(
        AnalyticalLinearDamper,
        damping_constant=damping_constant,
        time_step=time_step,
    )

    drag_enabled = (
        fluid_density is not None
        or drag_coeff_per is not None
        or drag_coeff_tan is not None
    )
    if drag_enabled:
        if fluid_density is None:
            raise ValueError("fluid_density must be set when enabling drag")
        if drag_step_skip <= 0:
            raise ValueError("drag_step_skip must be positive")
        if drag_callback_params is None:
            drag_callback_params = {}
        if drag_coeff_per is None or drag_coeff_tan is None:
            dl = L0 / n_elem
            r_bar = 0.5 * (radius_base + radius_tip)
            default_c_per = 0.41 / fluid_density / r_bar / dl
            default_c_tan = 0.033 / fluid_density / np.pi / r_bar / dl
            if drag_coeff_per is None:
                drag_coeff_per = default_c_per
            if drag_coeff_tan is None:
                drag_coeff_tan = default_c_tan
        from gym_softrobot.utils.actuation.forces.drag_force import DragForce

        simulator.add_forcing_to(rod).using(
            DragForce,
            rho_environment=fluid_density,
            c_per=drag_coeff_per,
            c_tan=drag_coeff_tan,
            system=rod,
            step_skip=drag_step_skip,
            callback_params=drag_callback_params,
        )

    return rod


def create_es_muscle_layers(
    radius_mean,
    radius_base,
):
    muscle_layers = [
        # LongitudinalMuscle(
        #     muscle_radius_ratio=np.stack(
        #         (np.zeros(radius_mean.shape),
        #          2 / 3 * np.ones(radius_mean.shape)),
        #         axis=0),
        #     max_force=1 * (radius_mean / radius_base) ** 2,
        # )
        LongitudinalMuscle(
            muscle_radius_ratio=np.stack(
                (np.zeros(radius_mean.shape), 6 / 9 * np.ones(radius_mean.shape)),
                axis=0,
            ),
            max_force=0.5 * (radius_mean / radius_base) ** 2,
        ),
        LongitudinalMuscle(
            muscle_radius_ratio=np.stack(
                (np.zeros(radius_mean.shape), -6 / 9 * np.ones(radius_mean.shape)),
                axis=0,
            ),
            max_force=0.5 * (radius_mean / radius_base) ** 2,
        ),
        TransverseMuscle(
            muscle_radius_ratio=np.stack(
                (np.zeros(radius_mean.shape), 4 / 9 * np.ones(radius_mean.shape)),
                axis=0,
            ),
            max_force=1.0 * (radius_mean / radius_base) ** 2,
        ),
    ]
    return muscle_layers
