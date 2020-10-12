# Anisotropic cardiac tissue in 3D

<img src=DATA/awesomenessity011.bmp height="500">

## This software was developerd by: **Hector Augusto Velasco-Perez** @CHAOS Lab@Georgia Institute of Technology

### Special thanks to:
- Dr. @Flavio Fenton
- Dr. @Claire Yanyan Ji
- Dr. @Abouzar Kaboudian
- Dr. @Shahriar Iravanian

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

## Software use (in order of appearance)
- To run the eprogram, open a Linux terminal and type `make`
- `globalVariables.cuh`:
     - `nx`,`ny`,`nz`: Grid size
     - `ITPERFRAME` is the number of iterations it computes in the background without rendering an image.
     - `Uth` is the filament voltage treshold.
     - `NN` is the size of the array in floats (doubles) to save the tip filament trajectory.
     - `DOUBLE_PRECISION`: switch between double and single presicion (comment/uncomment).
     - `ANISOTROPIC_TISSUE`: switch between isotropic and anisotropic fibers (comment/uncomment).
     - `LOAD_DATA`: switch from loading an external file or initializing the simulation with a precoded initial condition (comment/uncomment).
     - `SAVE_DATA`: switch between saving data or not saving data (comment/uncomment).
     - `SPIRAL_TIP_INTERPOLATION`: Switch between subpixel resolution interpolating algorithm or pixel resolution algorithm for filament tracing (comment/uncomment).
     - Model parameters. Remember to modify the parameters for the floating point type selected.
- `main3V-FK.cu`: 
     - `strAdress`: memory adress where the output data will be saved.
     - The structure `param` contains most of the physical parameters. To see which parameters it contains see `typedef3V-FK.h`.
     - `pwdAdress` contains the input-memory adress (if `LOAD_DATA` in not commented).
     - `keyboard` function: keyboard shortcuts. All names are self explanetory.
     - `exitProgram` function: comment/uncomment the functions inside the `#ifdef SAVE_DATA`.
