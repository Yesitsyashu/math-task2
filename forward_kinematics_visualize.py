#!/usr/bin/env python3
"""
forward_kinematics_visualize.py

Compute forward kinematics for a 4-DOF revolute robot (each link length L, default 1.0).
Visualize the robot in 3D and optionally save a PNG or an animation (GIF/MP4).

Conventions:
 - Joint order: j1, j2, j3, j4 (angles supplied in degrees by default)
 - Joint axes:
     j1 -> rotation about Z
     j2 -> rotation about Y
     j3 -> rotation about Y
     j4 -> rotation about X
 - For each joint: apply rotation about the joint axis, then translate along local X by link length L
 - Final transform: T_total = T1 * T2 * T3 * T4
 - End-effector position is translation part of T_total
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# -----------------------
# Linear algebra helpers
# -----------------------
def rot_x(theta):
    """4x4 homogeneous rotation about X (theta in radians)"""
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[1,0,0,0],
                     [0,c,-s,0],
                     [0,s, c,0],
                     [0,0,0,1]], dtype=float)

def rot_y(theta):
    """4x4 homogeneous rotation about Y (theta in radians)"""
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[ c,0,s,0],
                     [ 0,1,0,0],
                     [-s,0,c,0],
                     [ 0,0,0,1]], dtype=float)

def rot_z(theta):
    """4x4 homogeneous rotation about Z (theta in radians)"""
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[c,-s,0,0],
                     [s, c,0,0],
                     [0, 0,1,0],
                     [0, 0,0,1]], dtype=float)

def trans_x(a):
    """4x4 homogeneous translation along X by a"""
    return np.array([[1,0,0,a],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1]], dtype=float)

# -----------------------
# Forward kinematics
# -----------------------
def forward_kinematics(joints_deg, L=1.0):
    """
    joints_deg: iterable of 4 angles in degrees [j1, j2, j3, j4]
    L: link length (float)
    returns: positions (5x3 np.array) joint positions including base and end-effector, T_end (4x4)
    """
    if len(joints_deg) != 4:
        raise ValueError("Expected 4 joint angles (j1 j2 j3 j4)")

    # convert to radians
    j = np.radians(np.asarray(joints_deg, dtype=float))
    j1, j2, j3, j4 = j

    # T1: Rz(j1) then Tx(L)
    T1 = rot_z(j1) @ trans_x(L)
    # T2: Ry(j2) then Tx(L)
    T2 = rot_y(j2) @ trans_x(L)
    # T3: Ry(j3) then Tx(L)
    T3 = rot_y(j3) @ trans_x(L)
    # T4: Rx(j4) then Tx(L)
    T4 = rot_x(j4) @ trans_x(L)

    Ts = [np.eye(4), T1, T1@T2, T1@T2@T3, T1@T2@T3@T4]
    positions = np.array([T[:3,3] for T in Ts], dtype=float)  # 5 x 3

    return positions, Ts[-1]

# -----------------------
# Plotting utilities
# -----------------------
def plot_robot(positions, ax=None, title="4-DOF Robot (Forward Kinematics)", show_axes=True):
    """
    positions: Nx3 array of joint positions (base,...,end-effector)
    returns: matplotlib Axes object
    """
    xs = positions[:,0]; ys = positions[:,1]; zs = positions[:,2]
    if ax is None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
    # plot links
    ax.plot(xs, ys, zs, marker='o', linewidth=2)
    # annotate joints
    for i,(x,y,z) in enumerate(positions):
        ax.text(x, y, z, f' J{i}', fontsize=9)
    # equal aspect
    max_range = np.ptp(np.concatenate([xs[:,None], ys[:,None], zs[:,None]], axis=1), axis=0).max()
    if max_range == 0:
        max_range = 1.0
    mid_x = (xs.max()+xs.min())/2
    mid_y = (ys.max()+ys.min())/2
    mid_z = (zs.max()+zs.min())/2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title)
    if show_axes:
        origin = np.array([0.0,0.0,0.0])
        ax.quiver(origin[0], origin[1], origin[2], 0.3, 0, 0, linewidth=1)
        ax.quiver(origin[0], origin[1], origin[2], 0, 0.3, 0, linewidth=1)
        ax.quiver(origin[0], origin[1], origin[2], 0, 0, 0.3, linewidth=1)
    return ax

# -----------------------
# Animation / save utilities
# -----------------------
def animate_rotation(positions, save_path=None, fps=20, duration_sec=6):
    """
    Create a rotating animation of the plotted robot by changing the azimuth angle.
    - positions: Nx3 joint positions
    - save_path: optional path to save (supports .gif or .mp4). If None, returns the animation object without saving.
    """
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    plot_robot(positions, ax=ax)

    # set a static elevation and vary azim
    def update(frame):
        az = frame
        ax.view_init(elev=20, azim=az)
        return fig,

    frames = int(fps * duration_sec)
    angles = np.linspace(0, 360, frames)
    anim = animation.FuncAnimation(fig, lambda fr: update(angles[fr]), frames=frames, interval=1000/fps)

    if save_path:
        ext = save_path.lower().split('.')[-1]
        try:
            if ext == 'gif':
                # PillowWriter or ImageMagick
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            else:
                # mp4 via ffmpeg
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='fk-visualizer'))
                anim.save(save_path, writer=writer)
        except Exception as e:
            print("Warning: saving animation failed:", e)
            print("You may need ffmpeg or pillow installed in your system.")
    else:
        return anim

# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="4-DOF Forward Kinematics Visualizer")
    parser.add_argument('--joints', nargs=4, type=float, default=[0,0,0,0],
                        help="Four joint angles in degrees: j1 j2 j3 j4 (default: 0 0 0 0)")
    parser.add_argument('--L', type=float, default=1.0, help="Link length L (default 1.0 m)")
    parser.add_argument('--save-png', type=str, default=None, help="Path to save a PNG snapshot (optional)")
    parser.add_argument('--save-anim', type=str, default=None, help="Path to save animation (optional): .gif or .mp4")
    parser.add_argument('--no-show', action='store_true', help="Do not show interactive window (useful when saving only)")
    return parser.parse_args()

def main():
    args = parse_args()
    positions, T_end = forward_kinematics(args.joints, L=args.L)
    print("Joint angles (deg):", args.joints)
    print("Joint positions (x,y,z):\n", np.array2string(positions, precision=4, separator=', '))
    print("End-effector position:", np.array2string(positions[-1], precision=4, separator=', '))
    # Plot
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    plot_robot(positions, ax=ax)

    if args.save_png:
        try:
            fig.savefig(args.save_png, bbox_inches='tight', dpi=200)
            print("Saved PNG to", args.save_png)
        except Exception as e:
            print("Failed to save PNG:", e)

    if args.save_anim:
        print("Creating animation (this may take a few seconds)...")
        try:
            animate_rotation(positions, save_path=args.save_anim)
            print("Saved animation to", args.save_anim)
        except Exception as e:
            print("Failed to create/save animation:", e)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == '__main__':
    main()
