
/*
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Developed by: Hector Augusto Velasco-Perez 
@ CHAOS Lab 
@ Georgia Institute of Technology
August 07/10/2019

Special thanks to:
Dr. Flavio Fenton
Dr. Claire Yanyan Ji
Dr. Abouzar Kaboudian

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
*/

#define nx (500)
#define ny (500)
#define nz (32)

#define BLOCK_LENGTH (256)

#define I2D(nx,i,j) (((nx)*(j)) + i)
#define I3D(nx,nxy,i,j,k) ( ((nxy)*(k)) + ((nx)*(j)) + i)

#define pi (3.14159265359)

// Iterations per second
#define ITPERFRAME (50)

// Macro for copy kernel
#define BLOCK_I (64)

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

#ifndef STEP
#define STEP(a,b) (a >= b)
#endif

#define Uth (0.5f)

// N is the maximum number of structs to insert
#define NN 1000000

// Comment the next line for single precision arithmetic
#define DOUBLE_PRECISION
#define ANISOTROPIC_TISSUE
//#define PERIODIC_Z
#define LOAD_DATA
#define SAVE_DATA
#define SPIRALTIP_INTERPOLATION

/*========================================================================
* Model parameters defined as macros
*
* These parameters can be sent from the matlab side if the need arrises
* with minimal effect on the performance of the code
*========================================================================
*/

#ifdef DOUBLE_PRECISION

  #define C_m     (1.0)

  #define g_fi    (2.4)
  #define tau_r   (50.0)
  #define tau_si  (44.84)
  #define tau_o   (8.3)
  #define tau_vp  (3.33)
  #define tau_v1n (1000.0)
  #define tau_v2n (19.2)
  #define tau_wp  (667.0)
  #define tau_wn  (11.0)
  #define theta_c     (0.13)
  #define theta_v     (0.055)
  #define u_csi   (0.85)
  #define tau_d   (C_m / g_fi) //Cm/g_fi
  #define K       (10.0)

#else

  #define C_m     (1.0f)

  #define g_fi    (2.4f)
  #define tau_r   (50.0f)
  #define tau_si  (44.84f)
  #define tau_o   (8.3f)
  #define tau_vp  (3.33f)
  #define tau_v1n (1000.0f)
  #define tau_v2n (19.2f)
  #define tau_wp  (667.0f)
  #define tau_wn  (11.0f)
  #define theta_c     (0.13f)
  #define theta_v     (0.055f)
  #define u_csi   (0.85f)
  #define tau_d   (C_m / g_fi) //Cm/g_fi
  #define K       (10.0f)


 #endif
