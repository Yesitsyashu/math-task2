# Forward Kinematics – 4 DOF Robot Arm

This project implements the forward kinematics for a simple 4-DOF robotic arm.  
Each joint is revolute and every link has a length of 1 meter.  
The joint axes are arranged so that each joint axis is perpendicular to the previous one.

## How to run

python3 forward_kinematics_visualize.py


This shows the robot in 3D with all joint angles set to zero.

## Using custom joint angles (in degrees)

python3 forward_kinematics_visualize.py --joints 30 -45 60 90


## Saving a snapshot

python3 forward_kinematics_visualize.py --save-png output.png --no-show


## Saving a short rotating animation

python3 forward_kinematics_visualize.py --save-anim robot.gif --no-show


## Requirements

- Python 3  
- numpy  
- matplotlib  
- (optional) pillow / ffmpeg for saving animations

## Notes

The script computes the position of each joint using basic rotation + translation matrices  
and plots the resulting robot shape in 3D.
