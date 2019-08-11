# Anisotropic cardiac tissue in 3D

## This software was developerd by: **Hector Augusto Velasco-Perez** @CHAOS Lab@Georgia Institute of Technology

### Special thanks to:
- Dr. Flavio Fenton
- Dr. Claire Yanyan Ji
- Dr. Abouzar Kaboudian
- Dr. Shahriar Iravanian

## Software general decription
This software allows you to solve the Fenton-Karma (FK) model with a diffusive coupling in a 3D domain with a constant rotating conducting anisotropy. The software allows for input/output files and real time graphics for user interactivity.

## Other features
- Time integration fist Euler method
- Spacial coupuling: forth order Laplacian
- Filament tracking: pixel and subpixel resolution
- Zero-flux boundary conditions and optional periodic boundary conditions at the top and bottom of the domain
- Switch between anisotropic and isotropic tissue. The anisotropi is a constant rotating anisotropy
- Switch between single and doubl precision

## Software requirements
- CUDA v7 or higher
- glew.h, glut.h, freeglut.h
- SOIL.h library (sudo apt-get install libsoil-dev)

## Software organization (by function)
- main3V-FK.cu
  * main function


## Software use

