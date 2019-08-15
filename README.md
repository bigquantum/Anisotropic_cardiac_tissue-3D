# Anisotropic cardiac tissue in 3D

## This software was developerd by: **Hector Augusto Velasco-Perez** @CHAOS Lab@Georgia Institute of Technology

### Special thanks to:
- Dr. Flavio Fenton
- Dr. Claire Yanyan Ji
- Dr. Abouzar Kaboudian
- Dr. Shahriar Iravanian

## Software general decription
This software allows you to solve the Fenton-Karma (FK) model with a diffusive coupling in a 3D domain with a constant rotating conducting anisotropy. The software allows for input/output files and real time graphics for user interactivity. This software is implemented in C/CUDA.

## Other features
- Time integration: first order explicit Euler method
- Spacial coupling: fourth order Laplacian with finite differences
- Filament tracking: pixel and subpixel resolution
- Zero-flux boundary conditions and optional periodic boundary conditions at the top and bottom of the domain
- Switch between anisotropic and isotropic tissue. The anisotropy is a constant rotating anisotropy
- Switch between single and double precision

## Software requirements
- CUDA v7 or higher
- glew.h, glut.h, freeglut.h
- SOIL.h library (sudo apt-get install libsoil-dev)

## Software organization (by function)
- main3V-FK.cu
  * main function: variable 
  * 


## Software use
- Open a Linux terminal and type 'make'
