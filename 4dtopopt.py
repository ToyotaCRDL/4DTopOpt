#
# 4DTopOpt: Source code for the 4D topology optimization of soft bodies
#
# Copyright [2024] [Toyota Central R&D Labs., Inc.]
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The portion of this code pertaining to the physical simulation of
# soft bodies using the material point method has been adapted from
# the Taichi sample code "mpm3d.py":
#
#     https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm3d.py,
#
# which is provided under the Apache 2.0 license.
# The original code has been modified for this project, including the replacement of
# the material model with a neo-Hookean hyperelastic solid model, as well as
# various minor additions and modifications.
#


import taichi as ti
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import os, sys, time

from numpy_ml.neural_nets.optimizers import Adam
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from PIL import Image
from pyevtk.hl import pointsToVTK
from pyevtk.vtk import VtkFile, VtkRectilinearGrid


ti.reset()

RANDOM_SEED = 1809
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

figfontsize = 14
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = figfontsize
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05
palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray']


real = ti.f64
USE_GPU = False
if USE_GPU:
    ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, fast_math=False, device_memory_GB=2, random_seed=RANDOM_SEED, debug=False)
else:
    ti.init(default_fp=real, arch=ti.cpu, flatten_if=True, fast_math=False, cpu_max_num_threads=4, random_seed=RANDOM_SEED, debug=False)

case_name = '4dtopopt'
folder = case_name


#
# Computation parameters
#
dim = 2  # spatial dimension; 2 or 3
n_particles = 0  # will be overwritten later
n_solid_particles = 0  # will be overwritten later
n_actuators = 4

L = 1.0  # side length of computation domain, m
H = 0.03  # ground height, m
n_grid = 100  # number of grid per dimension
dx = L / n_grid
inv_dx = 1 / dx
particle_density = 2
bound_def = 3

block_size = 0.1  # side length of design domain, m
block_offset = (0.1, H, 0.5-block_size/2)  # for walker
#block_offset = (0.5-block_size/2, H, 0.5-block_size/2)  # for rotator

dt = 1e-4
max_steps = 5000  # total simulation time = max_steps * dt, s
steps = max_steps

# Learning rate; Default value for Adam is 0.001
lr = 0.01

# Filtering
filter_radius = dx / particle_density * 3. + 1e-10  # 0.015 m
filter_power = 2

# Projection
beta_lay = 4.
beta_topol = 4.

# Continuation; we do not use continuation here
beta_lay_max = beta_lay * 2**0
beta_topol_max = beta_topol * 2**0

# Augmented Lagrangian related parameters
la_l2_pen = 0.
la_quad_pen = 0.
la_pw_sgn_pen = 0.
la_pw_abs_pen = 0.
sigma_l2_pen = 0.001
sigma_quad_pen = 0.001
sigma_pw_sgn_pen = 0.001
sigma_pw_abs_pen = 0.001
stat_eps = 0.001
K_l2_pen_prev = (1. - (1./(n_actuators+1)**2) * (n_actuators+1)) * 0.95  # 95% of max
l2_tol = (1. - (1./(n_actuators+1)**2) * (n_actuators+1)) * 0.05
K_quad_pen_prev = 0.25 * 0.95
K_pw_sgn_pen_prev = 1. * 0.99  # 99% of max
K_pw_abs_pen_prev = 0.25 * 0.99

round_digit = 8

gif_fps = 200
SAVE_PNG = 0
SAVE_VTK = 0


#
# Physical parameters
#
p_vol = (dx / particle_density)**dim  # particle volume, m^dim
p_rho = 1e3  # particle density, kg/(m^dim)
p_mass = p_vol * p_rho  # kg
E = 1e5  # Young's modulus, Pa
nu = 0.4  # Poisson's ratio
mu = E / (2 * (1 + nu))  # first Lame parameter (mu)
la = E * nu / ((1 + nu) * (1 - 2 * nu))  # second Lame parameter (lambda)

matd = 1. / E  # Young's modulus at rho=0 is 1 Pa 
penal_mass = 1
penal_E = 1
penal_act = 3

gravity = 9.8
coeff_fric = -1  # default 0.5; static friction if < 0 

act_strength = 1e4  # max actuation strength, Pa
pulse_amp = 0.2
pulse_sd = 0.01
pulse_dt = 0.002
pulse_steps = int((max_steps * dt + 1e-10) / pulse_dt)


scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
vec_f32 = lambda: ti.Vector.field(dim, dtype=ti.f32)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
C, F = mat(), mat()
x_viz = vec_f32()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
loss = scalar()

actuation = scalar()
pulse_w_sgn, pulse_w_abs = scalar(), scalar()
pulse_w_sgn_proj, pulse_w_abs_proj = scalar(), scalar()

vec_n_act = lambda: ti.Vector.field(n_actuators+1, dtype=real)
act_w = vec_n_act()
act_w_filtered = vec_n_act()
act_w_soft = vec_n_act()

filter_index = scalar()
filter_weight = scalar()
n_filter_particles = 0  # will be overwritten later

is_design = ti.field(ti.i32)
design_id = ti.field(ti.i32)
design_id_inv = ti.field(ti.i32)
n_design_particles = 0  # will be overwritten later

rho = scalar()
phi = scalar()
phi_filtered = scalar()

n_penalty = 4
act_w_l2 = scalar()
rho_quad = scalar()
pulse_w_sgn_quad = scalar()
pulse_w_abs_quad = scalar()

total_mass = scalar()  # total mass of the particles
v_cg_step_sum = vec()
v_cg_step = vec()  # mass-weighted velocity (stepwise)
v_timesum = vec()  # distance over simulation time (-)
#x_cg_step_sum = vec()     # for rotator
#x_cg_step = vec()         # for rotator; center of gravity (stepwise)
#ang_mmt_step = vec()      # for rotator; angular momentum (stepwise, kg m(2) s(−1))
#mmt_iner_step = scalar()  # for rotator; moment of inertia (stepwise, kg m(2))
#ang_vel_step = vec()      # for rotator; angular velocity (stepwise, s(-1))
#ang_timesum = vec()       # for rotator; rotation angle over simulation time (-)

bound_x_max = scalar()
bound_x_min = scalar()
bound_y_max = scalar()
bound_y_min = scalar()
if dim == 3:
    bound_z_max = scalar()
    bound_z_min = scalar()

#bound_y_min_v = vec()

color_viz = ti.Vector.field(3, dtype=ti.f32)
density_viz = scalar()
if dim == 2:
    ground_vertices = ti.Vector.field(dim, dtype=ti.f32, shape=2)
    ground_indices = ti.field(ti.i32, shape=2)
elif dim == 3:
    ground_vertices = ti.Vector.field(dim, dtype=ti.f32, shape=2 * 2)
    ground_indices = ti.field(ti.i32, shape=2 * 3)
else:
    print('WARNING: Wrong dimension dim = {} is given'.format(dim))

x_sliced = vec()
act_viz = scalar()


def allocate_fields():
    ti.root.dense(ti.k, max_steps+1).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.l, n_particles).place(x_viz, color_viz, density_viz, x_sliced)
    if dim == 2:
        ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    elif dim == 3:
        ti.root.dense(ti.ijk, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, total_mass, v_timesum)

    ti.root.dense(ti.i, max_steps+1).place(v_cg_step_sum, v_cg_step)
    #ti.root.dense(ti.i, max_steps+1).place(x_cg_step_sum, x_cg_step)  # for rotator
    #ti.root.dense(ti.i, max_steps+1).place(ang_mmt_step, mmt_iner_step, ang_vel_step)  # for rotator
    #ti.root.place(ang_timesum)  # for rotator

    ti.root.dense(ti.i, pulse_steps+1).dense(ti.j, n_actuators).place(pulse_w_sgn, pulse_w_abs, pulse_w_sgn_proj, pulse_w_abs_proj)
    ti.root.dense(ti.i, max_steps+1).dense(ti.j, n_actuators).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)

    ti.root.dense(ti.i, n_particles).place(act_w, act_w_filtered, act_w_soft, act_viz)
    ti.root.dense(ti.i, n_design_particles).place(rho, phi, phi_filtered, design_id_inv)
    ti.root.dense(ti.i, n_particles).place(is_design, design_id)

    ti.root.dense(ti.i, n_particles).dense(ti.j, n_filter_particles).place(filter_index, filter_weight)
    
    ti.root.place(act_w_l2, rho_quad, pulse_w_sgn_quad, pulse_w_abs_quad)

    if dim == 2:
        ti.root.dense(ti.i, n_grid).place(bound_x_max, bound_x_min, bound_y_max, bound_y_min)
    elif dim == 3:
        ti.root.dense(ti.ij, n_grid).place(bound_x_max, bound_x_min, bound_y_max, bound_y_min, bound_z_max, bound_z_min)
        #ti.root.place(bound_y_min_v)

    ti.root.lazy_grad()


@ti.kernel
def clear_grid_2d():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0.0] * dim
        grid_m_in[i, j] = 0.0
        grid_v_in.grad[i, j] = [0.0] * dim
        grid_m_in.grad[i, j] = 0.0
        grid_v_out.grad[i, j] = [0.0] * dim


@ti.kernel
def clear_grid_3d():
    for i, j, k in grid_m_in:
        grid_v_in[i, j, k] = [0.0] * dim
        grid_m_in[i, j, k] = 0.0
        grid_v_in.grad[i, j, k] = [0.0] * dim
        grid_m_in.grad[i, j, k] = 0.0
        grid_v_out.grad[i, j, k] = [0.0] * dim


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0.0] * dim
        v.grad[f, i] = [0.0] * dim
        C.grad[f, i] = [[0.0] * dim] * dim
        F.grad[f, i] = [[0.0] * dim] * dim


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(0, n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=dim, val=1.) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()

        if particle_type[p] == 0:  # if fluid
            cbrtJ = ti.pow(J, 1/3)
            new_F = ti.Matrix.diag(dim=dim, val=cbrtJ)

        F[f + 1, p] = new_F

        act = 0.
        if actuator_id[p] == 0:
            for i in ti.static(range(n_actuators)):
                act += act_w_soft[p][i] * actuation[f, i] * act_strength

        if particle_type[p] == 0:  # if fluid
            act = 0.

        simp_mass = 1.
        simp_E = 1.
        simp_act = 1.
        if is_design[p] == 1:
            des_id = design_id[p]
            simp_mass = (1. - matd) * rho[des_id]**penal_mass + matd
            simp_E = (1. - matd) * rho[des_id]**penal_E + matd
            simp_act = (1. - matd) * rho[des_id]**penal_act + matd

        A = ti.Matrix.diag(dim=dim, val=1.) * act * simp_act
        cauchy = ti.Matrix([[0.0] * dim] * dim)
        mass = 0.0
        ident = ti.Matrix.diag(dim=dim, val=1.)

        mass = p_mass * simp_mass
        cauchy = simp_E * ( mu * (new_F @ new_F.transpose()) + ident * (la * ti.log(J) - mu) )  # neo-hookean

        if particle_type[p] == 0:  # if fluid
            mass = p_mass
            cauchy = ident * (J - 1) * E

        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        if ti.static(dim == 2):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                    weight = w[i][0] * w[j][1]
                    ti.atomic_add(grid_v_in[base + offset], 
                        weight * (mass * v[f, p] + affine @ dpos))
                    ti.atomic_add(grid_m_in[base + offset], (weight * mass))
        else:
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                        weight = w[i][0] * w[j][1] * w[k][2]
                        ti.atomic_add(grid_v_in[base + offset], 
                            weight * (mass * v[f, p] + affine @ dpos))
                        ti.atomic_add(grid_m_in[base + offset], (weight * mass))


@ti.kernel
def grid_op_2d():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity

        if i < bound_x_min[j] and v_out[0] < 0:  # x_min
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([1., 0.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        if i > bound_x_max[j] and v_out[0] > 0:  # x_max
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([-1., 0.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        if j < bound_y_min[i] and v_out[1] < 0:  # y_min
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([0., 1.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        if j > bound_y_max[i] and v_out[1] > 0:  # y_max
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([0., -1.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        grid_v_out[i, j] = v_out


@ti.kernel
def grid_op_3d():
    for i, j, k in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j, k] + 1e-10)
        v_out = inv_m * grid_v_in[i, j, k]
        v_out[1] -= dt * gravity

        if i < bound_x_min[j, k] and v_out[0] < 0:  # x_min
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([1., 0., 0.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        if i > bound_x_max[j, k] and v_out[0] > 0:  # x_max
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([-1., 0., 0.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        if j < bound_y_min[i, k] and v_out[1] < 0:  # y_min
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([0., 1., 0.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        if j > bound_y_max[i, k] and v_out[1] > 0:  # y_max
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([0., -1., 0.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        if k < bound_z_min[i, j] and v_out[2] < 0:  # z_min
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([0., 0., 1.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        if k > bound_z_max[i, j] and v_out[2] > 0:  # z_max
            if ti.static(coeff_fric < 0):
                v_out = ti.Vector([0.] * dim)
            else:
                normal = ti.Vector([0., 0., -1.])
                lin = (v_out.transpose() @ normal)(0)
                if lin < 0:
                    vit = v_out - lin * normal
                    lit = vit.norm() + 1e-10
                    if lit + coeff_fric * lin <= 0:
                        v_out = ti.Vector([0.] * dim)
                    else:
                        v_out = (1 + coeff_fric * lin / lit) * vit

        grid_v_out[i, j, k] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(0, n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0] * dim)
        new_C = ti.Matrix([[0.0] * dim] * dim)

        if ti.static(dim == 2):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j]), real) - fx
                    g_v = grid_v_out[base[0] + i, base[1] + j]
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        else:
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                        g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                        weight = w[i][0] * w[j][1] * w[k][2]
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def clear_pulse_actuation(t: ti.i32):
    for j in range(n_actuators):
        actuation[t, j] = 0.


@ti.kernel
def apply_projection_to_pulse_weight():
    for i, j in ti.ndrange(pulse_steps+1, n_actuators):
        pulse_w_sgn_proj[i, j] = pulse_w_sgn[i, j]              # [-1, 1] |-> [-1, 1]
        pulse_w_abs_proj[i, j] = pulse_w_abs[i, j] * 0.5 + 0.5  # [-1, 1] |-> [0, 1]


@ti.kernel
def apply_side_constraint_to_pulse_weight(lb: real, ub: real):
    for i, j in ti.ndrange(pulse_steps+1, n_actuators):
        pulse_w_sgn[i, j] = ti.max(lb, pulse_w_sgn[i, j])  # Side constraints [lb, ub]
        pulse_w_sgn[i, j] = ti.min(ub, pulse_w_sgn[i, j])
        pulse_w_abs[i, j] = ti.max(lb, pulse_w_abs[i, j])
        pulse_w_abs[i, j] = ti.min(ub, pulse_w_abs[i, j])


@ti.kernel
def compute_pulse_actuation_1(t: ti.i32):
    for i, j in ti.ndrange(pulse_steps+1, n_actuators):
        act = pulse_w_sgn_proj[i, j] * pulse_w_abs_proj[i, j] * \
              pulse_amp * ti.exp( -(ti.cast(t, real) * dt - ti.cast(i, real) * pulse_dt)**2 / (2. * pulse_sd**2) )
        ti.atomic_add(actuation[t, j], act)


@ti.kernel
def compute_pulse_actuation_2(t: ti.i32):
    for j in range(n_actuators):
        act = actuation[t, j]
        actuation[t, j] = ti.tanh(act)


@ti.kernel
def update_boundary(t: ti.i32):
    for i in range(n_grid):
        bound_y_min_v[None] = ti.Vector([0., 0., 0.03 * act_omega * ti.cos(act_omega * t * dt)])  # z-wise sine wave


@ti.kernel
def compute_total_mass():  # Compute total mass of design object
    for i in range(n_particles):
        contrib = p_mass
        if is_design[i] == 1:
            simp_mass = ((1.-matd)*rho[design_id[i]]**penal_mass + matd)
            contrib = simp_mass * p_mass
        ti.atomic_add(total_mass[None], contrib)


@ti.kernel
def clear_x_cg_step():
    for i in range(max_steps+1):
        x_cg_step_sum[i] = [0.0] * dim
        x_cg_step[i] = [0.0] * dim


@ti.kernel
def compute_x_cg_step_sum(f: ti.i32):
    for i in range(n_particles):
        contrib = p_mass
        if is_design[i] == 1:
            simp_mass = ((1. - matd) * rho[design_id[i]]**penal_mass + matd)
            contrib = p_mass * simp_mass
        ti.atomic_add(x_cg_step_sum[f+1], contrib * x[f+1, i])
        if f == 0:
            ti.atomic_add(x_cg_step_sum[0], contrib * x[0, i])


@ti.kernel
def compute_x_cg_step(f: ti.i32):
    x_cg_step[f+1] = x_cg_step_sum[f+1] / total_mass[None]
    if f == 0:
        x_cg_step[0] = x_cg_step_sum[0] / total_mass[None]


@ti.kernel
def clear_v_cg_step():
     for i in range(max_steps+1):
        v_cg_step_sum[i] = [0.0] * dim
        v_cg_step[i] = [0.0] * dim


@ti.kernel
def compute_v_cg_step_sum(f: ti.i32):
    for i in range(n_particles):
        contrib = p_mass
        if is_design[i] == 1:
            simp_mass = ((1. - matd) * rho[design_id[i]]**penal_mass + matd)
            contrib = p_mass * simp_mass
        ti.atomic_add(v_cg_step_sum[f+1], contrib * v[f+1, i])
        if f == 0:
            ti.atomic_add(v_cg_step_sum[0], contrib * v[0, i])


@ti.kernel
def compute_v_cg_step(f: ti.i32):
    if f == 0:
        v_cg_step[0] = v_cg_step_sum[0] / total_mass[None]
    v_cg_step[f+1] = v_cg_step_sum[f+1] / total_mass[None]


@ti.kernel
def clear_ang_vel_step():
     for i in range(max_steps+1):
        ang_mmt_step[i] = [0.0] * dim
        mmt_iner_step[i] = 0.0
        ang_vel_step[i] = [0.0] * dim


@ti.kernel
def compute_ang_mmt_step(f: ti.i32):
    for i in range(n_particles):
        contrib = p_mass
        if is_design[i] == 1:
            simp_mass = ((1. - matd) * rho[design_id[i]]**penal_mass + matd)
            contrib = p_mass * simp_mass
        x_cross_v = ( x[f+1, i] - x_cg_step[f+1] ).cross( v[f+1, i] - v_cg_step[f+1] )
        x_dot_x = ( x[f+1, i] - x_cg_step[f+1] ).norm()**2
        ti.atomic_add(ang_mmt_step[f+1], contrib * x_cross_v)
        ti.atomic_add(mmt_iner_step[f+1], contrib * x_dot_x)
        if f == 0:
            x_cross_v = ( x[0, i] - x_cg_step[0] ).cross( v[0, i] - v_cg_step[0] )
            x_dot_x = ( x[0, i] - x_cg_step[0] ).norm()**2
            ti.atomic_add(ang_mmt_step[0], contrib * x_cross_v)
            ti.atomic_add(mmt_iner_step[0], contrib * x_dot_x)


@ti.kernel
def compute_ang_vel_step(f: ti.i32):
    ang_vel_step[f+1] = ang_mmt_step[f+1] / mmt_iner_step[f+1]
    if f == 0:
        ang_vel_step[0] = ang_mmt_step[0] / mmt_iner_step[0]


@ti.kernel
def compute_v_timesum():
     for i in range(max_steps+1):
        contrib = dt
        ti.atomic_add(v_timesum[None], contrib * v_cg_step[i])


@ti.kernel
def compute_ang_timesum():
     for i in range(max_steps+1):
        contrib = dt
        ti.atomic_add(ang_timesum[None], contrib * ang_vel_step[i])


@ti.kernel
def compute_rho_quad():
    for p in range(n_particles):
        if is_design[p] == 1:
            rho_tmp = rho[design_id[p]]
            ti.atomic_add(rho_quad[None], rho_tmp * (1. - rho_tmp) / n_design_particles)  # quadratic penalty


@ti.kernel
def compute_act_w_l2():
    for p in range(n_particles):
        w_l2_sum = 0.
        if is_design[p] == 1:
            for i in ti.static(range(n_actuators+1)):
                w_l2_sum += act_w_soft[p][i]**2
            ti.atomic_add(act_w_l2[None], (1. - w_l2_sum) / n_design_particles)


@ti.kernel
def compute_pulse_w_quad():  # pulse actuator
    for i, j in ti.ndrange(pulse_steps+1, n_actuators):
        contrib = 1. / ((pulse_steps+1) * n_actuators)
        w_sgn = pulse_w_sgn_proj[i, j]
        w_abs = pulse_w_abs_proj[i, j]
        ti.atomic_add(pulse_w_sgn_quad[None], contrib * (1. + w_sgn) * (1. - w_sgn))  # quadratic penalty, max=1
        ti.atomic_add(pulse_w_abs_quad[None], contrib * w_abs * (1. - w_abs))  # quadratic penalty, max=0.25


@ti.kernel
def compute_loss(lag_coeffs: ti.types.matrix(n_penalty, 2, real)):
    dist_x = v_timesum[None][0]
    #dist_y = v_timesum[None][1]
    #ang_timesum_y = ang_timesum[None][1]  # for rotator

    pens = ti.Vector([
        ti.max((act_w_l2[None] - l2_tol), 0.),        # pen_xi, 5% tolerance
        ti.max((rho_quad[None] - 0.0125), 0.),        # pen_phi, 5% tolerance
        ti.max((pulse_w_sgn_quad[None] - 0.01), 0.),  # pen_pw_sgn, 1% tolerance
        ti.max((pulse_w_abs_quad[None] - 0.0025), 0.)   # pen_pw_abs, 1% tolerance
    ])

    loss[None] = -dist_x 
    #loss[None] = ang_timesum_y  # for rotator

    # Augmented Lagrangian method
    # >> lag_coeffs[i, 0]: lambda (kappa) for i-th constraint
    # >> lag_coeffs[i, 1]: sigma (tau) for i-th constraint
    for i in ti.static(range(n_penalty)):     
        ti.atomic_add(loss[None], - lag_coeffs[i, 0] * pens[i] + 0.5 * lag_coeffs[i, 1] * pens[i]**2) 


@ti.kernel
def make_ground_line():
    ground_vertices[0] = [0.0, H/L]
    ground_vertices[1] = [L/L, H/L]
    ground_indices[0] = 0
    ground_indices[1] = 1


@ti.kernel
def make_ground_mesh():
    ground_vertices[0] = [0.0, H, 0.0]
    ground_vertices[1] = [L, H, 0.0]
    ground_vertices[2] = [L, H, L]
    ground_vertices[3] = [0.0, H, L]
    ground_indices[0] = 0
    ground_indices[1] = 1
    ground_indices[2] = 2
    ground_indices[3] = 0
    ground_indices[4] = 2
    ground_indices[5] = 3


def make_animation(folder, duration=100, filename='movie'):
    global max_steps, gif_fps
    pictures=[]
    for s in range(0, max_steps, gif_fps):
#    for s in range(steps//sub_steps):
#        for i in range(sub_steps):
#            if (sub_steps*s + i) % int(1/(gif_fps*dt)) == 0:
#                glob_step = int(sub_steps*s + i)
        img = Image.open(f'{folder}/image/{s:05d}.png')
        pictures.append(img)

    pictures[0].save(f'{folder}/{filename}.gif', save_all=True, append_images=pictures[1:], optimize=False, duration=duration, loop=0)


def forward(total_steps, lag_coeffs, iter):
    total_mass[None] = 0.0
    compute_total_mass()
    clear_v_cg_step()
    #clear_x_cg_step()       # for rotator
    #clear_ang_vel_step()    # for rotator

    # simulation
    for s in range(total_steps):
        if dim == 2:
            clear_grid_2d()
        elif dim == 3:
            clear_grid_3d()
        clear_pulse_actuation(s)
        compute_pulse_actuation_1(s)
        compute_pulse_actuation_2(s)
        #update_boundary(s)
        p2g(s)
        if dim == 2:
            grid_op_2d()
        elif dim == 3:
            grid_op_3d()
        g2p(s)

        compute_v_cg_step_sum(s)
        compute_v_cg_step(s)
        #compute_x_cg_step_sum(s)  # for rotator
        #compute_x_cg_step(s)      # for rotator
        #compute_ang_mmt_step(s)   # for rotator
        #compute_ang_vel_step(s)   # for rotator

        if s % int(1/(gif_fps*dt)) == 0:
            particle_viz(s)
            render()
            if SAVE_PNG:
                filepath = folder+'/iter{:04d}'.format(iter)
                os.makedirs(filepath + '/image', exist_ok=True)
                window.save_image(filepath+'/image/{:05d}'.format(s)+'.png')
            window.show()

    if SAVE_PNG:
        try:
            make_animation(filepath)
        except:
            print('Error in gif generation')

    v_timesum[None] = [0.] * dim
    compute_v_timesum()
    #ang_timesum[None] = [0.] * dim  # for rotator
    #compute_ang_timesum()           # for rotator

    act_w_l2[None] = 0
    compute_act_w_l2()
    rho_quad[None] = 0
    compute_rho_quad()
    pulse_w_sgn_quad[None] = 0
    pulse_w_abs_quad[None] = 0
    compute_pulse_w_quad()
    compute_loss(lag_coeffs)

    return loss[None]


def backward(total_steps, lag_coeffs, iter):
    clear_particle_grad()

    compute_loss.grad(lag_coeffs)
    compute_pulse_w_quad.grad()
    compute_rho_quad.grad()
    compute_act_w_l2.grad()

    #compute_ang_timesum.grad()  # for rotator
    compute_v_timesum.grad()

    for s in reversed(range(total_steps)):
        # Since we do not store the grid history (to save space), we redo p2g and grid_op
        if dim == 2:
            clear_grid_2d()
        elif dim == 3:
            clear_grid_3d()
        p2g(s)
        if dim == 2:
            grid_op_2d()
        elif dim == 3:
            grid_op_3d()

        #compute_ang_vel_step.grad(s)   # for rotator
        #compute_ang_mmt_step.grad(s)   # for rotator
        #compute_x_cg_step.grad(s)      # for rotator
        #compute_x_cg_step_sum.grad(s)  # for rotator
        compute_v_cg_step.grad(s)
        compute_v_cg_step_sum.grad(s)

        g2p.grad(s)
        if dim == 2:
            grid_op_2d.grad()
        elif dim == 3:
            grid_op_3d.grad()
        p2g.grad(s)
        #update_boundary.grad(s)
        compute_pulse_actuation_2.grad(s)
        compute_pulse_actuation_1.grad(s)

        if SAVE_VTK:
            filepath = folder+'/iter{:04d}'.format(iter)
            os.makedirs(filepath + '/vtk', exist_ok=True)
            if s % int(1/(gif_fps*dt)) == 0:
                export_step_vtk(s, filepath+'/vtk/particle{:05d}'.format(s))
            if s == (total_steps - 1):  # if final time step
                clear_pulse_actuation(s + 1)
                compute_pulse_actuation_1(s + 1)
                compute_pulse_actuation_2(s + 1)
                export_step_vtk(s + 1, filepath+'/vtk/particle{:05d}'.format(s + 1))

    compute_total_mass.grad()


@ti.kernel
def clear_phi_filtered():
    for i in range(n_particles):
        if is_design[i] == 1:
            phi_filtered[design_id[i]] = 0.


def p2p_filter_topol(BACKWARD=False):
    if BACKWARD:
        p2p_filter_topol_seg_grad()
    else:
        p2p_filter_topol_seg()


@ti.kernel
def p2p_filter_topol_seg():
    for i, j in filter_index:
        if is_design[i] == 1:
            index = ti.cast(filter_index[i, j], ti.i32)
            weight = filter_weight[i, j]
            if index != -1:
                ti.atomic_add(phi_filtered[design_id[i]], weight * phi[design_id[index]])


@ti.kernel
def p2p_filter_topol_seg_grad():
    for i, j in filter_index:
        if is_design[i] == 1:
            index = ti.cast(filter_index[i, j], ti.i32)
            weight = filter_weight[i, j]
            if index != -1:
                ti.atomic_add(phi.grad[design_id[index]], weight * phi_filtered.grad[design_id[i]])


@ti.kernel
def apply_sigmoid_topol(beta_topol: real):
    for i in range(n_particles):
        if is_design[i] == 1:
            rho[design_id[i]] = 0.5*( ti.tanh(beta_topol*phi_filtered[design_id[i]]) / ti.tanh(beta_topol) + 1. )  # [-1, 1] |-> [0, 1]


@ti.kernel
def apply_side_constraint_to_phi(lb: real, ub: real):
    for i in range(n_design_particles):
        phi[i] = ti.max(lb, phi[i])  # Side constraints for phi
        phi[i] = ti.min(ub, phi[i])


@ti.kernel
def clear_act_w_filtered():
    for i in range(n_particles):
        for j in ti.static(range(n_actuators+1)):
            act_w_filtered[i][j] = 0.


def p2p_filter_lay(BACKWARD=False):
    if BACKWARD:
        p2p_filter_lay_seg_grad()
    else:
        p2p_filter_lay_seg()


@ti.kernel
def p2p_filter_lay_seg():
    for i, j in filter_index:
        index = ti.cast(filter_index[i, j], ti.i32)
        weight = filter_weight[i, j]
        if index != -1:
            ti.atomic_add(act_w_filtered[i], weight * act_w[index])


@ti.kernel
def p2p_filter_lay_seg_grad():
    for i, j in filter_index:
        index = ti.cast(filter_index[i, j], ti.i32)
        weight = filter_weight[i, j]
        if index != -1:
            ti.atomic_add(act_w.grad[index], weight * act_w_filtered.grad[i])


@ti.kernel
def apply_softmax_lay(beta_lay: real):
    for i in range(n_particles):
        exp_sum = 0.
        for j in ti.static(range(n_actuators+1)):
            exp_sum += ti.exp(beta_lay * act_w_filtered[i][j])
        for j in ti.static(range(n_actuators+1)):
            act_w_soft[i][j] = ti.exp(beta_lay * act_w_filtered[i][j]) / exp_sum


def compute_p2p_filter_weight(x, isd, radius, power=3):
    global dim, n_grid, inv_dx, n_particles, n_filter_particles
    x = np.asarray(x)
    isd = np.asarray(isd)
    if dim == 2:
        grid_list = [[[] for i in range(n_grid)] for i in range(n_grid)]
    elif dim == 3:
        grid_list = [[[[] for i in range(n_grid)] for i in range(n_grid)] for i in range(n_grid)]
    index_list = [[] for i in range(n_particles)]
    weight_list = [[] for i in range(n_particles)]
    grid_radius = np.ceil(radius * inv_dx).astype(int)

    print('Computing filter weights for ({} x {})...'.format(n_particles, n_particles))
    time_start = time.time()

    if dim == 2:
        for p in range(n_particles):
            base = np.floor(x[p, :] * inv_dx).astype(int)
            grid_list[base[0]][base[1]].append(p)
    elif dim == 3:
        for p in range(n_particles):
            base = np.floor(x[p, :] * inv_dx).astype(int)
            grid_list[base[0]][base[1]][base[2]].append(p)

    n_neighbor_max = 0
    if dim == 2:
        for ps in range(n_particles):
            if isd[ps] == 0: continue
            base = np.floor(x[ps, :] * inv_dx).astype(int)
            n_neighbor = 0
            weight_sum = 0
            for i in range(-grid_radius, grid_radius):
                for j in range(-grid_radius, grid_radius):
                    if not 0 <= base[0]+i <= n_grid-1: continue
                    if not 0 <= base[1]+j <= n_grid-1: continue
                    for pe in grid_list[base[0]+i][base[1]+j]:
                        if isd[pe] == 0: continue
                        dist = np.linalg.norm(x[pe, :] - x[ps, :], ord=2)  # distance from particle ps to pe
                        if dist <= radius:
                            weight = (1. - min(dist, radius)/radius)**power
                            index_list[ps].append(pe)
                            weight_list[ps].append(weight)
                            n_neighbor += 1
                            weight_sum += weight
            n_neighbor_max = max(n_neighbor_max, n_neighbor)
            if not weight_sum == 0.:
                weight_list[ps] = [(weight / weight_sum) for weight in weight_list[ps]]
    elif dim == 3:
        for ps in range(n_particles):
            if isd[ps] == 0: continue
            base = np.floor(x[ps, :] * inv_dx).astype(int)
            n_neighbor = 0
            weight_sum = 0
            for i in range(-grid_radius, grid_radius):
                for j in range(-grid_radius, grid_radius):
                    for k in range(-grid_radius, grid_radius):
                        if not 0 <= base[0]+i <= n_grid-1: continue
                        if not 0 <= base[1]+j <= n_grid-1: continue
                        if not 0 <= base[2]+k <= n_grid-1: continue
                        for pe in grid_list[base[0]+i][base[1]+j][base[2]+k]:
                            if isd[pe] == 0: continue
                            dist = np.linalg.norm(x[pe, :] - x[ps, :], ord=2)  # distance from particle ps to pe
                            if dist <= radius:
                                weight = (1. - min(dist, radius)/radius)**power
                                index_list[ps].append(pe)
                                weight_list[ps].append(weight)
                                n_neighbor += 1
                                weight_sum += weight
            n_neighbor_max = max(n_neighbor_max, n_neighbor)
            if not weight_sum == 0.:
                weight_list[ps] = [(weight / weight_sum) for weight in weight_list[ps]]

    index_np = np.full(shape=(n_particles, n_neighbor_max), fill_value=-1)
    weight_np = np.zeros(shape=(n_particles, n_neighbor_max))
    for p in range(n_particles):
        for k in range(len(index_list[p])):
            index_np[p, k] = index_list[p][k]
            weight_np[p, k] = weight_list[p][k]

    n_filter_particles = n_neighbor_max

    print('  >> Done.')
    print('  >> Max. no. particles in filter radius: {}'.format(n_neighbor_max))
    print('  >> Elapsed time: {:.3f} s'.format(time.time()-time_start))

    return index_np, weight_np


@ti.kernel
def particle_viz(f: ti.i32):
    for i in range(n_particles):
        if ti.static(dim == 2):
            x_viz[i] = x[f, i] / L
        else:
            if is_design[i] and rho[design_id[i]] < 0.5:
                x_viz[i] = [0.5, -0.1, 0.5]
            else:
                x_viz[i] = x[f, i]

        act = 0.
        if actuator_id[i] == 0:
            for j in ti.static(range(n_actuators)):
                act += act_w_soft[i][j] * actuation[f, j]  # interval: [-1, 1]
            color_viz[i] = [0.5 - act, 0.5 - abs(act), 0.5 + act]
        else:
            color_viz[i] = [0., 0., 0.]

        if is_design[i] == 1:
            density_viz[i] = rho[design_id[i]]
        else:
            density_viz[i] = 1.0

        # Apply transparency to colors
        color_bg = ti.Vector([1., 1., 1.])
        r, g, b, a = color_viz[i][0], color_viz[i][1], color_viz[i][2], density_viz[i]
        R, G, B = color_bg
        color_viz[i][0] = r * a + (1. - a) * R
        color_viz[i][1] = g * a + (1. - a) * G
        color_viz[i][2] = b * a + (1. - a) * B


@ti.kernel
def actuation_viz(f: ti.i32):
    for p in range(0, n_particles):
        act_viz[p] = 0.
        if actuator_id[p] == 0:
            for i in ti.static(range(n_actuators)):
                act_viz[p] += act_w_soft[p][i] * actuation[f, ti.max(0, i)] * act_strength
        if particle_type[p] == 0:  # if fluid
            act_viz[p] = 0.


def export_step_vtk(s, filename):
    global n_particles
    x_np = np.zeros(shape=(n_particles, 3))
    x_np[:, 0:dim] = x.to_numpy()[s]
    is_design_np = is_design.to_numpy()
    design_id_inv_np = design_id_inv.to_numpy()

    rho_np = np.full(n_particles, np.nan)
    rho_np[is_design_np == 0] = 1.  # rho = 1 for non-design domain
    rho_np[design_id_inv_np] = rho.to_numpy()
    #rho_grad_np = np.full(n_particles, np.nan)
    #rho_grad_np[design_id_inv_np] = rho.grad.to_numpy()  # note: phi.grad is before backprop
    phi_np = np.full(n_particles, np.nan)
    phi_np[is_design_np == 0] = 1.  # phi = 1 for non-design domain
    phi_np[design_id_inv_np] = phi.to_numpy()

    actuator_id_np = actuator_id.to_numpy()
    actuation_viz(s)
    actuation_np = act_viz.to_numpy()
    act_w_soft_argmax_np = np.argmax(act_w_soft.to_numpy(), axis=1)

    pointsToVTK(filename, x_np[:, 0].copy(), x_np[:, 1].copy(), x_np[:, 2].copy(), data={
        "rho": rho_np.copy(), "phi": phi_np.copy(),
        #"dLdrho": rho_grad_np.copy(),
        "actuator_id": actuator_id_np.copy(), "actuation": actuation_np.copy(),
        "actuator_no": act_w_soft_argmax_np.copy()
    })
    # the reason of using .copy() -> see https://vtk.org/pipermail/vtk-developers/2010-November/024428.html


def export_iter_vtk(filename):
    global n_particles
    x_np = np.zeros(shape=(n_particles, 3))
    x_np[:, 0:dim] = x.to_numpy()[0]
    is_design_np = is_design.to_numpy()
    design_id_inv_np = design_id_inv.to_numpy()

    rho_np = np.full(n_particles, np.nan)
    rho_np[is_design_np == 0] = 1.  # rho = 1 for non-design domain
    rho_np[design_id_inv_np] = rho.to_numpy()
    #rho_grad_np = np.full(n_particles, np.nan)
    #rho_grad_np[design_id_inv_np] = rho.grad.to_numpy()
    phi_np = np.full(n_particles, np.nan)
    phi_np[is_design_np == 0] = 1.  # phi = 1 for non-design domain
    phi_np[design_id_inv_np] = phi.to_numpy()
    #phi_grad_np = np.full(n_particles, np.nan)
    #phi_grad_np[design_id_inv_np] = phi.grad.to_numpy()

    actuator_id_np = actuator_id.to_numpy()
    act_w_soft_argmax_np = np.argmax(act_w_soft.to_numpy(), axis=1)

    pointsToVTK(filename, x_np[:, 0].copy(), x_np[:, 1].copy(), x_np[:, 2].copy(), data={
        "rho": rho_np.copy(), "phi": phi_np.copy(),
    #    "dLdrho": rho_grad_np.copy(), "dLdphi": phi_grad_np.copy(),
        "actuator_id": actuator_id_np.copy(),
        "actuator_no": act_w_soft_argmax_np.copy()
    })
    # the reason of .copy() -> see https://vtk.org/pipermail/vtk-developers/2010-November/024428.html


def sig_round(x, digit):
    x_order = np.round(np.log10(np.abs(x)+1e-15), 0)
    return np.round(x*np.power(10, -x_order), digit-1)*np.power(10, x_order)


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.n_design_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.is_design = []
        self.design_id = []
        self.design_id_inv = []
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
        self.num_actuators = 0

    def new_actuator(self):
        self.num_actuators += 1
        global n_actuators
        n_actuators = self.num_actuators
        return self.num_actuators - 1

    def add_rect(self, x, y, w, h, actuation, ptype=1, is_design=0):
        if ptype == 0:
            assert actuation == -1
        global n_particles, particle_density
        w_count = int(w / dx * particle_density)
        h_count = int(h / dx * particle_density)
        real_dx = w / w_count
        real_dy = h / h_count

        if ptype == 1: # solid
            for i in range(w_count):
                for j in range(h_count):
                    self.x.append([
                        x + (i + 0.5) * real_dx + self.offset_x,
                        y + (j + 0.5) * real_dy + self.offset_y
                    ])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.is_design.append(is_design)
                    self.design_id.append(self.n_design_particles)
                    if is_design:
                        self.design_id_inv.append(self.n_particles)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)
                    self.n_design_particles += int(is_design == 1)
        else: # fluid
            for i in range(w_count):
                for j in range(h_count):
                    self.x.append([
                        x + random.random() * w + self.offset_x,
                        y + random.random() * h + self.offset_y
                    ])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)

    def add_cubo(self, x, y, z, w, h, d, actuation, ptype=1, is_design=0):
        if ptype == 0:
            assert actuation == -1
        global n_particles, particle_density
        w_count = int(w / dx * particle_density)
        h_count = int(h / dx * particle_density)
        d_count = int(d / dx * particle_density)
        real_dx = w / w_count
        real_dy = h / h_count
        real_dz = d / d_count

        if ptype == 1: # solid
            for i in range(w_count):
                for j in range(h_count):
                    for k in range(d_count):
                        self.x.append([
                            x + (i + 0.5) * real_dx + self.offset_x,
                            y + (j + 0.5) * real_dy + self.offset_y,
                            z + (k + 0.5) * real_dz + self.offset_z
                        ])
                        self.actuator_id.append(actuation)
                        self.particle_type.append(ptype)
                        self.is_design.append(is_design)
                        self.design_id.append(self.n_design_particles)
                        if is_design:
                            self.design_id_inv.append(self.n_particles)
                        self.n_particles += 1
                        self.n_solid_particles += int(ptype == 1)
                        self.n_design_particles += int(is_design == 1)
        else: # fluid
            for i in range(w_count):
                for j in range(h_count):
                    for k in range(d_count):
                        self.x.append([
                            x + random.random() * w + self.offset_x,
                            y + random.random() * h + self.offset_y,
                            z + random.random() * d + self.offset_z
                        ])
                        self.actuator_id.append(actuation)
                        self.particle_type.append(ptype)
                        self.n_particles += 1
                        self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y, z=0.):
        self.offset_x = x
        self.offset_y = y
        self.offset_z = z

    def finalize(self):
        global n_particles, n_solid_particles, n_design_particles
        n_particles = self.n_particles
        n_solid_particles = max(self.n_solid_particles, 1)
        n_design_particles = max(self.n_design_particles, 1)
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)
        print('n_design', n_design_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = max(n_act, 1)


def design_walker_2d(scene):
    scene.set_offset(0.1, 0.03)
    scene.add_rect(0., 0., 0.2, 0.2, actuation=0, is_design=1)


def design_climber_2d(scene):
    scene.set_offset(0.4, 0.1)
    scene.add_rect(0., 0., 0.2, 0.2, actuation=0, is_design=1)


def design_balancer_2d(scene):
    scene.set_offset(0.45, 0.03)
    scene.add_rect(0., 0., 0.1, 0.4, actuation=0, is_design=1)
    scene.add_rect(0., -0.01, 0.1, 0.01, actuation=1, is_design=0)    # root
    scene.add_rect(0.045, 0.4, 0.01, 0.01, actuation=9, is_design=0)  # tip


def design_cube(scene):
    global block_size, block_offset
    scene.set_offset(block_offset[0], block_offset[1], block_offset[2])
    scene.add_cubo(0., 0., 0., block_size, block_size, block_size, actuation=0, is_design=1)


class Boundary:
    def __init__(self, n_grid, default_bound=3):
        global dim
        self.n_grid = n_grid
        self.viz_lines = []
        if dim == 2:
            self.x_max = np.full(shape=n_grid, fill_value=n_grid-default_bound)  # x at (y)
            self.x_min = np.full(shape=n_grid, fill_value=default_bound)
            self.y_max = np.full(shape=n_grid, fill_value=n_grid-default_bound)  # y at (x)
            self.y_min = np.full(shape=n_grid, fill_value=default_bound)
        elif dim == 3:
            self.x_max = np.full(shape=(n_grid, n_grid), fill_value=n_grid-default_bound)  # x at (y, z)
            self.x_min = np.full(shape=(n_grid, n_grid), fill_value=default_bound)
            self.y_max = np.full(shape=(n_grid, n_grid), fill_value=n_grid-default_bound)  # y at (x, z)
            self.y_min = np.full(shape=(n_grid, n_grid), fill_value=default_bound)
            self.z_max = np.full(shape=(n_grid, n_grid), fill_value=n_grid-default_bound)  # z at (x, y)
            self.z_min = np.full(shape=(n_grid, n_grid), fill_value=default_bound)
    
    def draw_line(self, which_b, point1, point2):
        #
        # Draw line between point1 and point2
        #     which_b: str; 'top', 'bottom', 'left', 'right' 
        #     point1, point2: two-dimensional list or ndarray with elements in [0, 1]
        #
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        if x1 != x2:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = (y2 - y1) / 1e-10
        intercept = y1 - slope * x1

        if which_b == 'top':
            b, l_type = self.b_t, 'h'
        elif which_b == 'bottom':
            b, l_type = self.b_b, 'h'
        elif which_b == 'left':
            b, l_type = self.b_l, 'v'
        elif which_b == 'right':
            b, l_type = self.b_r, 'v'
        else:
            print("ERROR: Wrong string is given for class Boundary.")

        if l_type == 'h':
            i_str = int(min(x1, x2) * n_grid)
            i_end = int(max(x1, x2) * n_grid)
            for i in range(i_str, i_end, 1):
                x = i / n_grid
                y = slope * x + intercept
                b[i] = y * n_grid
        if l_type == 'v':
            i_str = int(min(y1, y2) * n_grid)
            i_end = int(max(y1, y2) * n_grid)
            for i in range(i_str, i_end, 1):
                y = i / n_grid
                x = (y - intercept) / slope
                b[i] = x * n_grid

        self.viz_lines.append([point1, point2])  # for visualization
        #print("Set line between ({}, {}) and ({}, {}) for bounds_{}".format(x1, y1, x2, y2, which_b))


def boundary_ground(boundary, height):
    global dim, n_grid, L
    if dim == 2:
        for i in range(n_grid):
            boundary.y_min[i] = height * n_grid / L
    elif dim == 3:
        for i in range(n_grid):
            for j in range(n_grid):
                boundary.y_min[i, j] = height * n_grid / L


def boundary_2d_walls(boundary):
    global n_grid, L
    # make x_max wall
    for i in range(n_grid):
        y = i / n_grid * L
        for j in range(n_grid):
            z = j / n_grid * L
            if y <= 0.5 and z <= 0.5:
                boundary.x_max[i, j] = 0.5 * n_grid / L
    # make z_min wall
    for i in range(n_grid):
        x = i / n_grid * L
        for j in range(n_grid):
            y = j / n_grid * L
            if x >= 0.5 and y <= 0.5:
                boundary.z_min[i, j] = 0.5 * n_grid / L
    # make y_min wall
    for i in range(n_grid):
        x = i / n_grid * L
        for j in range(n_grid):
            z = j / n_grid * L
            if x >= 0.5 and z <= 0.5:
                boundary.y_min[i, j] = 0.5 * n_grid / L


window = ti.ui.Window("Diffmpm", (800, 800), vsync=False)
canvas = window.get_canvas()
if dim == 3:
    scene_ = ti.ui.Scene()
    camera = ti.ui.make_camera()


def render():
    if dim == 2:
        canvas.set_background_color(color=(1., 1., 1.))
        canvas.circles(x_viz, radius=(dx/particle_density/2), per_vertex_color=color_viz)
        make_ground_line()
        canvas.lines(ground_vertices, width=0.006*L, indices=ground_indices, color=(0.0, 0.0, 0.0))
        #for i in range(len(boundary.viz_lines)):
        #    gui.line(boundary.viz_lines[i][0], boundary.viz_lines[i][1], radius=6, color=0x0)
    elif dim == 3:
        camera.position(0.2 * L, 1.0 * L, 2 * L)
        camera.lookat(0.5 * L, 0.5 * L, 0.5 * L)
        scene_.set_camera(camera)
        scene_.ambient_light((0, 0, 0))
        scene_.point_light(pos=(0.5, 1.5, 0.5), color=(0.7, 0.7, 0.7))
        scene_.point_light(pos=(-0.5, 1.5, 1.5), color=(0.7, 0.7, 0.7))
        scene_.point_light(pos=(0.5, -1.5, 0.5), color=(0.7, 0.7, 0.7))
        scene_.particles(x_viz, radius=0.005*L, per_vertex_color=color_viz)
        make_ground_mesh()
        scene_.mesh(ground_vertices, ground_indices, color=(0.5, 0.5, 0.5))
        canvas.scene(scene_)


def save_des_var_as_csv(folder, iter, pulse_w_sgn, pulse_w_abs, actuation, act_w, act_w_soft, rho, phi):  # pulse actuator
    os.makedirs(folder, exist_ok=True)
    np.savetxt(f'{folder}/pulse_w_sgn_iter{iter:04d}.csv', pulse_w_sgn.to_numpy(), delimiter=',')
    np.savetxt(f'{folder}/pulse_w_abs_iter{iter:04d}.csv', pulse_w_abs.to_numpy(), delimiter=',')
    np.savetxt(f'{folder}/actuation_iter{iter:04d}.csv', actuation.to_numpy(), delimiter=',')
    np.savetxt(f'{folder}/act_w_iter{iter:04d}.csv', act_w.to_numpy(), delimiter=',')
    np.savetxt(f'{folder}/act_w_soft_iter{iter:04d}.csv', act_w_soft.to_numpy(), delimiter=',')
    np.savetxt(f'{folder}/rho_iter{iter:04d}.csv', rho.to_numpy(), delimiter=',')
    np.savetxt(f'{folder}/phi_iter{iter:04d}.csv', phi.to_numpy(), delimiter=',')


def load_des_var_from_csv(folder, iter):
    pulse_w_sgn_filename = folder + '/pulse_w_sgn_iter{:04d}.csv'.format(iter)
    pulse_w_abs_filename = folder + '/pulse_w_abs_iter{:04d}.csv'.format(iter)
    act_w_filename = folder + '/act_w_iter{:04d}.csv'.format(iter)
    phi_filename = folder + '/phi_iter{:04d}.csv'.format(iter)
    pulse_w_sgn.from_numpy(np.loadtxt(pulse_w_sgn_filename, delimiter=','))
    pulse_w_abs.from_numpy(np.loadtxt(pulse_w_abs_filename, delimiter=','))
    act_w.from_numpy(np.loadtxt(act_w_filename, delimiter=','))
    phi.from_numpy(np.loadtxt(phi_filename, delimiter=','))


def save_des_var_as_pickle(folder, iter, pulse_w_sgn, pulse_w_abs, act_w, phi):
    os.makedirs(folder, exist_ok=True)
    filename = f'{folder}/iter{iter:04d}.pickle'
    with open (filename, 'wb') as f:
        pickle.dump({'pulse_w_sgn': pulse_w_sgn.to_numpy(), 'pulse_w_abs': pulse_w_abs.to_numpy(),
                     'act_w': act_w.to_numpy(), 'phi': phi.to_numpy()}, f)


def load_des_var_from_pickle(folder, iter):
    filename = f'{folder}/iter{iter:04d}.pickle'
    with open(filename, 'rb') as f:
        des_vars = pickle.load(f)
        pulse_w_sgn.from_numpy(des_vars['pulse_w_sgn'])
        pulse_w_abs.from_numpy(des_vars['pulse_w_abs'])
        act_w.from_numpy(des_vars['act_w'])
        phi.from_numpy(des_vars['phi'])


def plot_actuation(steps, actuation):
    global dt, n_actuators, act_strength, palette
    t_local = np.zeros(shape=steps+1)
    act_local = np.zeros(shape=(steps+1, n_actuators))

    compute_pulse_actuation_1(steps)
    compute_pulse_actuation_2(steps)

    for s in range(steps+1):
        t_local[s] = s * dt
        for n in range(n_actuators):
            act_local[s, n] = actuation[s, n] * act_strength

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    for n in range(n_actuators):
        ax.plot(t_local, act_local[:, n], label='actuator {}'.format(n+1),
                marker='o', markersize=0, color=palette[n], linestyle='solid', linewidth=2, clip_on=False)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Actuation (Pa)')
    ax.set_xlim([0., steps*dt])
    ax.set_ylim([-act_strength*1.0, act_strength*1.0])
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.show();


def plot_actuation_w_pulse(steps, actuation, pulse_w_sgn, pulse_w_abs):
    global dt, n_actuators, act_strength, palette, pulse_dt, pulse_steps
    t_local = np.zeros(shape=steps+1)
    act_local = np.zeros(shape=(steps+1, n_actuators))
    t_pulse_local = np.zeros(shape=pulse_steps+1)
    pulse_w_local = np.zeros(shape=(pulse_steps+1, n_actuators))

    compute_pulse_actuation_1(steps)
    compute_pulse_actuation_2(steps)

    for s in range(pulse_steps+1):
        t_pulse_local[s] = s * pulse_dt
        for n in range(n_actuators):
            pulse_w_local[s, n] = pulse_w_sgn[s, n] * (0.5 * pulse_w_abs[s, n] + 0.5) * \
                                  act_strength * 0.1

    for s in range(steps+1):
        t_local[s] = s * dt
        for n in range(n_actuators):
            act_local[s, n] = actuation[s, n] * act_strength

    for n in range(n_actuators):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1)
        markerline, stemline, baseline, = ax.stem(t_pulse_local, pulse_w_local[:, n])
        plt.setp(baseline, linewidth=0.)
        plt.setp(stemline, linewidth=0.5, color='black')
        plt.setp(markerline, markersize=0)
        ax.plot(t_local, act_local[:, n], label='actuator {}'.format(n+1),
                marker='o', markersize=0, color=palette[n], linestyle='solid', linewidth=2, clip_on=False)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Actuation (Pa)')
        ax.set_xlim([0., steps*dt])
        ax.set_ylim([-act_strength*1.0, act_strength*1.0])
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.show();


def main():
    global SAVE_PNG, SAVE_VTK
    global beta_lay, beta_topol, beta_lay_max, beta_topol_max
    global la_l2_pen, la_quad_pen, la_pw_sgn_pen, la_pw_abs_pen
    global sigma_l2_pen, sigma_quad_pen, sigma_pw_sgn_pen, sigma_pw_abs_pen
    global stat_eps, l2_tol, K_l2_pen_prev, K_quad_pen_prev, K_pw_sgn_pen_prev, K_pw_abs_pen_prev

    #
    # initialization
    #
    scene = Scene()
    design_walker_2d(scene)
    #design_cube(scene)
    scene.finalize()
    boundary = Boundary(n_grid, bound_def)
    boundary_ground(boundary, H)
    #boundary_walls(boundary)

    # compute filter index and weight
    index, weight = compute_p2p_filter_weight(scene.x, scene.is_design, radius=filter_radius, power=filter_power)

    allocate_fields()

    bound_x_max.from_numpy(boundary.x_max)
    bound_x_min.from_numpy(boundary.x_min)
    bound_y_max.from_numpy(boundary.y_max)
    bound_y_min.from_numpy(boundary.y_min)
    if dim == 3:
        bound_z_max.from_numpy(boundary.z_max)
        bound_z_min.from_numpy(boundary.z_min)

    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = np.identity(dim).tolist()
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
        is_design[i] = scene.is_design[i]
        design_id[i] = scene.design_id[i]

    for i in range(n_design_particles):
        design_id_inv[i] = scene.design_id_inv[i]

    for i in range(pulse_steps):
        for j in range(n_actuators):
            pulse_w_sgn[i, j] = np.random.normal(loc=0, scale=0.1)  # N(0, 0.1^2)
            pulse_w_abs[i, j] = np.random.normal(loc=0, scale=0.1)  # N(0, 0.1^2)
    apply_side_constraint_to_pulse_weight(-1., 1.)

    for i in range(n_particles):
        for j in range(n_actuators+1):
            act_w[i][j] = 0.
    for i in range(n_design_particles):
        phi[i] = 0.


    filter_index.from_numpy(index)
    filter_weight.from_numpy(weight)  # is taichi field
    del index, weight
    
    losses = []
    l_tasks = []
    l_l2_pens = []
    l_l2_pens_tol = []
    l_quad_pens = []
    l_quad_pens_tol = []
    l_pw_sgn_pens = []
    l_pw_sgn_pens_tol = []
    l_pw_abs_pens = []
    l_pw_abs_pens_tol = []
    
    lag_sigma_l2_pens = []
    lag_sigma_quad_pens = []
    lag_sigma_pw_sgn_pens = []
    lag_sigma_pw_abs_pens = []
    lag_la_l2_pens = []
    lag_la_quad_pens = []
    lag_la_pw_sgn_pens = []
    lag_la_pw_abs_pens = []

    optimizer = Adam(lr=lr)

    iter = 0
    grace_iter = 0.
    STOP_ITER = False
    CONVERGED = False


    #
    # Main loop for optimization
    #
    while not STOP_ITER:

        if iter % 50 == 0:
            SAVE_PNG = 1
            SAVE_VTK = 1
        else:
            SAVE_PNG = 0
            SAVE_VTK = 0

        if CONVERGED:  # or iter == 2000
            gif_fps = 200
            SAVE_PNG = 1
            SAVE_VTK = 1
            STOP_ITER = True

        t_istart = time.time()
        ti.ad.clear_all_gradients()

        clear_act_w_filtered()
        p2p_filter_lay()
        apply_softmax_lay(beta_lay)
        clear_phi_filtered()
        p2p_filter_topol()
        apply_sigmoid_topol(beta_topol)
        apply_projection_to_pulse_weight()

        lag_coeffs = np.asarray([
            [la_l2_pen, sigma_l2_pen], [la_quad_pen, sigma_quad_pen], [la_pw_sgn_pen, sigma_pw_sgn_pen], [la_pw_abs_pen, sigma_pw_abs_pen]
        ])
        l = forward(max_steps, lag_coeffs, iter)


        # Note: Last few digits are truncated using sig_round() for the result reproducibility
        losses.append(sig_round(l, round_digit))
        l_tasks.append(sig_round(-v_timesum[None][0], round_digit))
        #l_tasks.append(sig_round(ang_timesum[None][1], round_digit))  # for rotator
        l_l2_pens.append(sig_round(act_w_l2[None], round_digit))
        l_l2_pens_tol.append(sig_round(max((act_w_l2[None] - l2_tol), 0.), round_digit))
        l_quad_pens.append(sig_round(rho_quad[None], round_digit))
        l_quad_pens_tol.append(sig_round(max((rho_quad[None] - 0.0125), 0.), round_digit))
        l_pw_sgn_pens.append(sig_round(pulse_w_sgn_quad[None], round_digit))
        l_pw_sgn_pens_tol.append(sig_round(max((pulse_w_sgn_quad[None] - 0.01), 0.), round_digit))
        l_pw_abs_pens.append(sig_round(pulse_w_abs_quad[None], round_digit))
        l_pw_abs_pens_tol.append(sig_round(max((pulse_w_abs_quad[None] - 0.0025), 0.), round_digit))
        lag_sigma_l2_pens.append(sig_round(sigma_l2_pen, round_digit))
        lag_la_l2_pens.append(sig_round(la_l2_pen, round_digit))
        lag_sigma_quad_pens.append(sig_round(sigma_quad_pen, round_digit))
        lag_la_quad_pens.append(sig_round(la_quad_pen, round_digit))
        lag_sigma_pw_sgn_pens.append(sig_round(sigma_pw_sgn_pen, round_digit))
        lag_la_pw_sgn_pens.append(sig_round(la_pw_sgn_pen, round_digit))
        lag_sigma_pw_abs_pens.append(sig_round(sigma_pw_abs_pen, round_digit))
        lag_la_pw_abs_pens.append(sig_round(la_pw_abs_pen, round_digit))

        ti.ad.clear_all_gradients()
        loss.grad[None] = 1
        backward(max_steps, lag_coeffs, iter)

        apply_projection_to_pulse_weight.grad()
        apply_sigmoid_topol.grad(beta_topol)
        p2p_filter_topol(BACKWARD=True)
        apply_softmax_lay.grad(beta_lay)
        p2p_filter_lay(BACKWARD=True)

        t_ielap = time.time() - t_istart
        print('i= {}, L= {:.5f}, D_x= {:.5f}, P_xi= {:.5f}, P_phi= {:.5f}, P_pws= {:.5f}, P_pwa = {:.5f}, t_elap= {:.1f} s'.format(
            iter, l, l_tasks[-1], l_l2_pens[-1], l_quad_pens[-1], l_pw_sgn_pens[-1], l_pw_abs_pens[-1], t_ielap
        ))

        # Output result files
        particle_viz(max_steps - 1)
        render()
        filepath = folder+'/hist'
        os.makedirs(filepath + '/image', exist_ok=True)
        window.save_image(filepath+'/image/image_iter{:04d}'.format(iter)+'.png')
        window.show()
        os.makedirs(filepath + '/vtk', exist_ok=True)
        export_iter_vtk(filepath+'/vtk/particle_iter{:04d}'.format(iter))
        save_des_var_as_pickle(filepath+'/des_var', iter, pulse_w_sgn, pulse_w_abs, act_w, phi)
        if SAVE_VTK:
            save_des_var_as_csv(folder+'/iter{:04d}/des_var'.format(iter), iter, pulse_w_sgn, pulse_w_abs, actuation, act_w, act_w_soft, rho, phi)  # pulse actuator


        # Augmented Lagrangian method (convergence check based on the loss; revised on 221101) 
        # Check if the solution is stationary based on the moving average of loss
        num_avg = 10
        if grace_iter > 2*num_avg and not CONVERGED:
            l_avg_cur = np.average(losses[-num_avg:])
            l_avg_prev = np.average(losses[-2*num_avg:-num_avg])
            l_rel_err = np.abs(l_avg_cur - l_avg_prev) / np.abs(l_avg_prev)

            if l_rel_err < stat_eps:  # if stationary
                print()
                print('loss is stationary. rel_err = {}'.format(l_rel_err))
                grace_iter = 0.
                optimizer = Adam(lr=lr)

                K_l2_pen = l_l2_pens_tol[-1]
                if K_l2_pen > 0.:  # if infeasible
                    if K_l2_pen < 0.25 * K_l2_pen_prev:
                        la_l2_pen -= sigma_l2_pen * K_l2_pen
                        K_l2_pen_prev = K_l2_pen
                        print('lambda_xi is updated to: {}, K = {}'.format(la_l2_pen, K_l2_pen))
                        print()
                    else:
                        sigma_l2_pen *= 10.
                        beta_lay = min(beta_lay*2., beta_lay_max)
                        print('sigma_xi is updated to: {}'.format(sigma_l2_pen))
                        print()
                else:
                    print('xi is feasible')

                K_quad_pen = l_quad_pens_tol[-1]
                if K_quad_pen > 0.:  # if infeasible
                    if K_quad_pen < 0.25 * K_quad_pen_prev:
                        la_quad_pen -= sigma_quad_pen * K_quad_pen
                        K_quad_pen_prev = K_quad_pen
                        print('lambda_phi is updated to: {}, K = {}'.format(la_quad_pen, K_quad_pen))
                        print()
                    else:
                        sigma_quad_pen *= 10.
                        beta_topol = min(beta_topol*2., beta_topol_max)
                        print('sigma_phi is updated to: {}'.format(sigma_quad_pen))
                        print()
                else:
                    print('phi is feasible')

                K_pw_sgn_pen = l_pw_sgn_pens_tol[-1]
                if K_pw_sgn_pen > 0.:  # if infeasible
                    if K_pw_sgn_pen < 0.25 * K_pw_sgn_pen_prev:
                        la_pw_sgn_pen -= sigma_pw_sgn_pen * K_pw_sgn_pen
                        K_pw_sgn_pen_prev = K_pw_sgn_pen
                        print('lambda_pw_sgn is updated to: {}, K = {}'.format(la_pw_sgn_pen, K_pw_sgn_pen))
                        print()
                    else:
                        sigma_pw_sgn_pen *= 10.
                        print('sigma_pw_sgn is updated to: {}'.format(sigma_pw_sgn_pen))
                        print()
                else:
                    print('pw_sgn is feasible')

                K_pw_abs_pen = l_pw_abs_pens_tol[-1]
                if K_pw_abs_pen > 0.:  # if infeasible
                    if K_pw_abs_pen < 0.25 * K_pw_abs_pen_prev:
                        la_pw_abs_pen -= sigma_pw_abs_pen * K_pw_abs_pen
                        K_pw_abs_pen_prev = K_pw_abs_pen
                        print('lambda_pw_abs is updated to: {}, K = {}'.format(la_pw_abs_pen, K_pw_abs_pen))
                        print()
                    else:
                        sigma_pw_abs_pen *= 10.
                        print('sigma_pw_abs is updated to: {}'.format(sigma_pw_abs_pen))
                        print()
                else:
                    print('pw_abs is feasible')

                if max(K_l2_pen, K_quad_pen, K_pw_sgn_pen, K_pw_abs_pen) == 0.:  # if all feasible
                    print()
                    print('Solution is converged.')
                    print()
                    CONVERGED = True


        if not CONVERGED:
            # Update design variables
            # >> We truncate the last few digits using sig_round() for the result reproducibility
            new_phi = optimizer.update(phi.to_numpy(), sig_round(phi.grad.to_numpy(), round_digit), 'phi')
            phi.from_numpy(new_phi)
            apply_side_constraint_to_phi(-1., 1.)

            new_act_w = optimizer.update(act_w.to_numpy(), sig_round(act_w.grad.to_numpy(), round_digit), 'act_w')
            act_w.from_numpy(new_act_w)

            new_pulse_w_sgn = optimizer.update(pulse_w_sgn.to_numpy(), sig_round(pulse_w_sgn.grad.to_numpy(), round_digit), 'pulse_w_sgn')
            pulse_w_sgn.from_numpy(new_pulse_w_sgn)
            new_pulse_w_abs = optimizer.update(pulse_w_abs.to_numpy(), sig_round(pulse_w_abs.grad.to_numpy(), round_digit), 'pulse_w_abs')
            pulse_w_abs.from_numpy(new_pulse_w_abs)
            apply_side_constraint_to_pulse_weight(-1., 1.)


        iter += 1
        grace_iter += 1


    # Save log as csv file
    os.makedirs('{}/log'.format(case_name), exist_ok=True)
    np.savetxt('{}/log/_losses.csv'.format(case_name), np.asarray(losses), delimiter=',')
    np.savetxt('{}/log/_l_tasks.csv'.format(case_name), np.asarray(l_tasks), delimiter=',')
    np.savetxt('{}/log/_l_l2_pens.csv'.format(case_name), np.asarray(l_l2_pens), delimiter=',')
    np.savetxt('{}/log/_l_quad_pens.csv'.format(case_name), np.asarray(l_quad_pens), delimiter=',')
    np.savetxt('{}/log/_l_pw_sgn_pens.csv'.format(case_name), np.asarray(l_pw_sgn_pens), delimiter=',')
    np.savetxt('{}/log/_l_pw_abs_pens.csv'.format(case_name), np.asarray(l_pw_abs_pens), delimiter=',')
    np.savetxt('{}/log/_lag_sigma_l2_pens.csv'.format(case_name), np.asarray(lag_sigma_l2_pens), delimiter=',')
    np.savetxt('{}/log/_lag_sigma_quad_pens.csv'.format(case_name), np.asarray(lag_sigma_quad_pens), delimiter=',')
    np.savetxt('{}/log/_lag_sigma_pw_sgn_pens.csv'.format(case_name), np.asarray(lag_sigma_pw_sgn_pens), delimiter=',')
    np.savetxt('{}/log/_lag_sigma_pw_abs_pens.csv'.format(case_name), np.asarray(lag_sigma_pw_abs_pens), delimiter=',')
    np.savetxt('{}/log/_lag_la_l2_pens.csv'.format(case_name), np.asarray(lag_la_l2_pens), delimiter=',')
    np.savetxt('{}/log/_lag_la_quad_pens.csv'.format(case_name), np.asarray(lag_la_quad_pens), delimiter=',')
    np.savetxt('{}/log/_lag_la_pw_sgn_pens.csv'.format(case_name), np.asarray(lag_la_pw_sgn_pens), delimiter=',')
    np.savetxt('{}/log/_lag_la_pw_abs_pens.csv'.format(case_name), np.asarray(lag_la_pw_abs_pens), delimiter=',')


if __name__ == '__main__':
    main()
