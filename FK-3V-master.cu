
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

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "typedef3V-FK.h"
//#include "globalVariables.cuh"
#include "hostPrototypes.h"
#include "devicePrototypes.cuh"

#include "./common/CudaSafeCall.h"

// Weight constants
extern __constant__ REAL dt_d, rx_d, ry_d, rz_d;
extern __constant__ REAL rxyc_d, rxzc_d, ryzc_d, rxyzf_d;
extern __constant__ REAL rCxyz_d, rwe_d, rsn_d, rbt_d;
extern __constant__ REAL rxy_d, rbx_d, rby_d;

// Miscellaneous constants
extern __constant__ REAL expTau_vp_d, expTau_wp_d, expTau_wn_d;
extern __constant__ REAL invdx_d, invdy_d, invdz_d;

extern __device__ vec3dyn dev_data1[NN];
extern __device__ vec6dyn dev_data2[NN];
extern __device__ int dev_count;

/*========================================================================
 * Main Entry of the Kernel
 *========================================================================
 */

 __global__ void FK_3V_kernel(stateVar g_out, stateVar g_in, conductionVar r,
  REAL *J_d) {

  /*------------------------------------------------------------------------
  * Getting i and j global indices
  *------------------------------------------------------------------------
  */

  const int i = threadIdx.x;
  const int j = blockIdx.x*blockDim.y + threadIdx.y;
  const int k = blockIdx.y;

  /*------------------------------------------------------------------------
  * return if we are outside the domain
  *------------------------------------------------------------------------
  */

  if( i >= nx && j>=ny && k >= nz) {
      return ;
  }

  /*------------------------------------------------------------------------
  * Converting global index into matrix indices assuming
  * the column major structure of the matlab matrices
  *------------------------------------------------------------------------
  */

  const int nxy = nx*ny;
  const int i3d = k * nxy + j * nx + i;

  /*------------------------------------------------------------------------
  * Setting local variables
  *------------------------------------------------------------------------
  */

  #ifdef DOUBLE_PRECISION

  REAL u = g_in.u[i3d] ;
  REAL v = g_in.v[i3d] ;
  REAL w = g_in.w[i3d] ;

  /*------------------------------------------------------------------------
  * Additional heaviside functions
  *------------------------------------------------------------------------
  */

  REAL  p     = ( u > theta_c   )   ? 1.0:0.0 ;
  REAL  q     = ( u > theta_v   )   ? 1.0:0.0 ;

  /*------------------------------------------------------------------------
  * Calculating dependant tau's
  *------------------------------------------------------------------------
  */

  REAL tau_vnu  =   q*tau_v1n + (1.0-q)*tau_v2n;

  g_out.v[i3d] =
    ( u > theta_c   ) ? v*expTau_vp_d : 1.0-(1.0-v)*exp(-dt_d/tau_vnu);

  g_out.w[i3d] =
    ( u > theta_c   ) ? w*expTau_wp_d : 1.0-(1.0-w)*expTau_wn_d;

/*
  REAL  dv2dt   =   (1.0-p)*(1.0-v)/tau_vnu - p*v/tau_vp;
  v += dv2dt*dt_d ;
  g_out.v[i3d] = v ;

  REAL dw2dt    =   (1.0-p)*(1.0-w)/tau_wn - p*w/tau_wp;
  w += dw2dt*dt_d ;
  g_out.w[i3d] = w ;
*/

  /*------------------------------------------------------------------------
  * I_sum
  *------------------------------------------------------------------------
  */

  //Fast inward  (Sodium)
  REAL  J_fi    =    -p*v*(1.0-u)*(u-theta_c)/tau_d;

  //Slow outward (Potassium)
  REAL  J_so    =    (1.0-p)*u/tau_o + p/tau_r;

  //Slow inward  (Calcium)
  REAL  J_si    =    -w*(1.0 + tanh(K*(u-u_csi)))/(2.0*tau_si);


  REAL  I_sum   =    J_fi + J_so + J_si ;

  J_d[i3d] = I_sum;

  /*------------------------------------------------------------------------
  * Laplacian Calculation
  *
  * No flux boundary condition is applied on all boundaries through
  * the Laplacian operator definition
  *------------------------------------------------------------------------
  */

  int S = ( j> 0      ) ? I3D(nx,nxy,i,j-1,k) : I3D(nx,nxy,i,j+1,k) ;
  int N = ( j<(ny-1)  ) ? I3D(nx,nxy,i,j+1,k) : I3D(nx,nxy,i,j-1,k) ;
  int W = ( i> 0      ) ? I3D(nx,nxy,i-1,j,k) : I3D(nx,nxy,i+1,j,k) ;
  int E = ( i<(nx-1)  ) ? I3D(nx,nxy,i+1,j,k) : I3D(nx,nxy,i-1,j,k) ;

  //////////////////////

  int SWxy = (i>0  && j>0) ? I3D(nx,nxy,i-1,j-1,k) :
            ((i==0 && j>0) ? I3D(nx,nxy,i+1,j-1,k) :
            ((i>0  && j==0)? I3D(nx,nxy,i-1,j+1,k) : I3D(nx,nxy,i+1,j+1,k) ) ) ;

  int SExy = (i<(nx-1)  && j>0) ? I3D(nx,nxy,i+1,j-1,k) :
            ((i==(nx-1) && j>0) ? I3D(nx,nxy,i-1,j-1,k) :
            ((i<(nx-1)  && j==0)? I3D(nx,nxy,i+1,j+1,k) : I3D(nx,nxy,i-1,j+1,k) ) ) ;

  int NWxy = (i>0  && j<(ny-1)) ? I3D(nx,nxy,i-1,j+1,k) :
            ((i==0 && j<(ny-1)) ? I3D(nx,nxy,i+1,j+1,k) :
            ((i>0  && j==(ny-1))? I3D(nx,nxy,i-1,j-1,k) : I3D(nx,nxy,i+1,j-1,k) ) ) ;

  int NExy = (i<(nx-1)  && j<(ny-1)) ? I3D(nx,nxy,i+1,j+1,k) :
            ((i==(nx-1) && j<(ny-1)) ? I3D(nx,nxy,i-1,j+1,k) :
            ((i<(nx-1)  && j==(ny-1))? I3D(nx,nxy,i+1,j-1,k) : I3D(nx,nxy,i-1,j-1,k) ) ) ;


 #ifdef PERIODIC_Z // In the z direction

  int B = ( k> 0      ) ? I3D(nx,nxy,i,j,k-1) : I3D(nx,nxy,i,j,nz-1) ;
  int T = ( k<(nz-1)  ) ? I3D(nx,nxy,i,j,k+1) : I3D(nx,nxy,i,j,0) ;

  int SWxz = (i>0  && k>0) ? I3D(nx,nxy,i-1,j,k-1) :
          ((i==0 && k>0) ? I3D(nx,nxy,i+1,j,k-1) :
          ((i>0  && k==0)? I3D(nx,nxy,i-1,j,nz-1) : I3D(nx,nxy,i+1,j,k+1) ) ) ;

  int SExz = (i<(nx-1)  && k>0) ? I3D(nx,nxy,i+1,j,k-1) :
            ((i==(nx-1) && k>0) ? I3D(nx,nxy,i-1,j,k-1) :
            ((i<(nx-1)  && k==0)? I3D(nx,nxy,i+1,j,nz-1) : I3D(nx,nxy,i-1,j,k+1) ) ) ;

  int NWxz = (i>0  && k<(nz-1)) ? I3D(nx,nxy,i-1,j,k+1) :
            ((i==0 && k<(nz-1)) ? I3D(nx,nxy,i+1,j,k+1) :
            ((i>0  && k==(nz-1))? I3D(nx,nxy,i-1,j,0) : I3D(nx,nxy,i+1,j,k-1) ) ) ;

  int NExz = (i<(nx-1)  && k<(nz-1)) ? I3D(nx,nxy,i+1,j,k+1) :
            ((i==(nx-1) && k<(nz-1)) ? I3D(nx,nxy,i-1,j,k+1) :
            ((i<(nx-1)  && k==(nz-1))? I3D(nx,nxy,i+1,j,0) : I3D(nx,nxy,i-1,j,k-1) ) ) ;

//////////////////////////

  int SWyz = (j>0  && k>0) ? I3D(nx,nxy,i,j-1,k-1) :
            ((j==0 && k>0) ? I3D(nx,nxy,i,j+1,k-1) :
            ((j>0  && k==0)? I3D(nx,nxy,i,j-1,nz-1) : I3D(nx,nxy,i,j+1,k+1) ) ) ;

  int SEyz = (j<(ny-1)  && k>0) ? I3D(nx,nxy,i,j+1,k-1) :
            ((j==(ny-1) && k>0) ? I3D(nx,nxy,i,j-1,k-1) :
            ((j<(ny-1)  && k==0)? I3D(nx,nxy,i,j+1,nz-1) : I3D(nx,nxy,i,j-1,k+1) ) ) ;

  int NWyz = (j>0  && k<(nz-1)) ? I3D(nx,nxy,i,j-1,k+1) :
            ((j==0 && k<(nz-1)) ? I3D(nx,nxy,i,j+1,k+1) :
            ((j>0  && k==(nz-1))? I3D(nx,nxy,i,j-1,0) : I3D(nx,nxy,i,j+1,k-1) ) ) ;

  int NEyz = (j<(ny-1)  && k<(nz-1)) ? I3D(nx,nxy,i,j+1,k+1) :
            ((j==(ny-1) && k<(nz-1)) ? I3D(nx,nxy,i,j-1,k+1) :
            ((j<(ny-1)  && k==(nz-1))? I3D(nx,nxy,i,j+1,0) : I3D(nx,nxy,i,j-1,k-1) ) ) ;

  #else            


  int B = ( k> 0      ) ? I3D(nx,nxy,i,j,k-1) : I3D(nx,nxy,i,j,k+1) ;
  int T = ( k<(nz-1)  ) ? I3D(nx,nxy,i,j,k+1) : I3D(nx,nxy,i,j,k-1) ;

  int SWxz = (i>0  && k>0) ? I3D(nx,nxy,i-1,j,k-1) :
            ((i==0 && k>0) ? I3D(nx,nxy,i+1,j,k-1) :
            ((i>0  && k==0)? I3D(nx,nxy,i-1,j,k+1) : I3D(nx,nxy,i+1,j,k+1) ) ) ;

  int SExz = (i<(nx-1)  && k>0) ? I3D(nx,nxy,i+1,j,k-1) :
            ((i==(nx-1) && k>0) ? I3D(nx,nxy,i-1,j,k-1) :
            ((i<(nx-1)  && k==0)? I3D(nx,nxy,i+1,j,k+1) : I3D(nx,nxy,i-1,j,k+1) ) ) ;

  int NWxz = (i>0  && k<(nz-1)) ? I3D(nx,nxy,i-1,j,k+1) :
            ((i==0 && k<(nz-1)) ? I3D(nx,nxy,i+1,j,k+1) :
            ((i>0  && k==(nz-1))? I3D(nx,nxy,i-1,j,k-1) : I3D(nx,nxy,i+1,j,k-1) ) ) ;

  int NExz = (i<(nx-1)  && k<(nz-1)) ? I3D(nx,nxy,i+1,j,k+1) :
            ((i==(nx-1) && k<(nz-1)) ? I3D(nx,nxy,i-1,j,k+1) :
            ((i<(nx-1)  && k==(nz-1))? I3D(nx,nxy,i+1,j,k-1) : I3D(nx,nxy,i-1,j,k-1) ) ) ;

//////////////////////////

  int SWyz = (j>0  && k>0) ? I3D(nx,nxy,i,j-1,k-1) :
            ((j==0 && k>0) ? I3D(nx,nxy,i,j+1,k-1) :
            ((j>0  && k==0)? I3D(nx,nxy,i,j-1,k+1) : I3D(nx,nxy,i,j+1,k+1) ) ) ;

  int SEyz = (j<(ny-1)  && k>0) ? I3D(nx,nxy,i,j+1,k-1) :
            ((j==(ny-1) && k>0) ? I3D(nx,nxy,i,j-1,k-1) :
            ((j<(ny-1)  && k==0)? I3D(nx,nxy,i,j+1,k+1) : I3D(nx,nxy,i,j-1,k+1) ) ) ;

  int NWyz = (j>0  && k<(nz-1)) ? I3D(nx,nxy,i,j-1,k+1) :
            ((j==0 && k<(nz-1)) ? I3D(nx,nxy,i,j+1,k+1) :
            ((j>0  && k==(nz-1))? I3D(nx,nxy,i,j-1,k-1) : I3D(nx,nxy,i,j+1,k-1) ) ) ;

  int NEyz = (j<(ny-1)  && k<(nz-1)) ? I3D(nx,nxy,i,j+1,k+1) :
            ((j==(ny-1) && k<(nz-1)) ? I3D(nx,nxy,i,j-1,k+1) :
            ((j<(ny-1)  && k==(nz-1))? I3D(nx,nxy,i,j+1,k-1) : I3D(nx,nxy,i,j-1,k-1) ) ) ;


 #endif


  #ifdef ANISOTROPIC_TISSUE

  /*------------------------------------------------------------------------
  * Anisotropic Laplacian
  *------------------------------------------------------------------------
  */

  REAL rx = r.x[k];
  REAL ry = r.y[k];
  REAL rz = r.z[k];
  REAL rbx = r.bx[k];
  REAL rby = r.by[k];

  REAL du2dt = (  rCxyz_d * (rx + ry + rz)*u
            +    rwe_d * (4.0*rx - ry - rz)*(g_in.u[W] + g_in.u[E])
            +    rsn_d * (4.0*ry - rx - rz)*(g_in.u[N] + g_in.u[S])
            +    rbt_d * (4.0*rz - ry - rx)*(g_in.u[T] + g_in.u[B])
            +    rxyc_d * (rx + ry)*(  g_in.u[SWxy] +
                                       g_in.u[SExy] +
                                       g_in.u[NWxy] +
                                       g_in.u[NExy] )
            +    rxzc_d * (rx + rz)*(  g_in.u[SWxz] +
                                       g_in.u[SExz] +
                                       g_in.u[NWxz] +
                                       g_in.u[NExz] )
            +    ryzc_d * (ry + rz)*(  g_in.u[SWyz] +
                                       g_in.u[SEyz] +
                                       g_in.u[NWyz] +
                                       g_in.u[NEyz] ) ) ;

  du2dt -= ( dt_d*( 0.5*I_sum
        +   rxyzf_d * ( ( J_d[E] + J_d[W] )
        +               ( J_d[N] + J_d[S] )
        +               ( J_d[B] + J_d[T] ) ) ) / C_m ) ;

/*
  REAL du2dt = (
    +    ( g_in.u[W] - 2.f*u + g_in.u[E] )*rx
    +    ( g_in.u[N] - 2.f*u + g_in.u[S] )*ry
    +    ( g_in.u[T] - 2.f*u + g_in.u[B] )*rz );

  du2dt -= dt_d*I_sum/C_m ;
*/

  // Correction to NSWE boundary conditions
  REAL b_S = (j > 0 )? 0.0:
            ((j==0 && (i==0 || i==(nx-1)))? 0.0:
            rby*(g_in.u[I3D(nx,nxy,i+1,j,k)] - g_in.u[I3D(nx,nxy,i-1,j,k)])) ;

  REAL b_N = (j < (ny-1))? 0.0:
            ((j==(ny-1) && (i==0 || i==(nx-1)))? 0.0:
            -rby*(g_in.u[I3D(nx,nxy,i+1,j,k)] - g_in.u[I3D(nx,nxy,i-1,j,k)])) ;

  REAL b_W = (i > 0 )? 0.0:
            ((i==0 && (j==0 || j==(ny-1)))? 0.0:
            rbx*(g_in.u[I3D(nx,nxy,i,j+1,k)] - g_in.u[I3D(nx,nxy,i,j-1,k)])) ;

  REAL b_E = (i < (nx-1))? 0.0:
            ((i==(nx-1) && (j==0 || j==(ny-1)))? 0.0:
            -rbx*(g_in.u[I3D(nx,nxy,i,j+1,k)] - g_in.u[I3D(nx,nxy,i,j-1,k)])) ;

  du2dt += (
           (  b_S + b_N )*ry
       +   (  b_W + b_E )*rx  );


  // Correcion to SW SE NW NE boundary conditions
  REAL b_SW = (i>0  && j>0)?  0.0 :
             ((i==0 && j>1)?  rbx*(g_in.u[i3d] - g_in.u[I3D(nx,nxy,i,j-2,k)]) :
             ((i>1  && j==0)? rby*(g_in.u[i3d] - g_in.u[I3D(nx,nxy,i-2,j,k)]) : 0.0)) ;

  REAL b_SE = (i<(nx-1)  && j>0)?  0.0 :
             ((i==(nx-1) && j>1)?  -rbx*(g_in.u[i3d] - g_in.u[I3D(nx,nxy,i,j-2,k)]) :
             ((i<(nx-2)  && j==0)? rby*(g_in.u[I3D(nx,nxy,i+2,j,k)] - g_in.u[i3d]) : 0.0)) ;

  REAL b_NW = (i>0  && j<(ny-1))?  0.0 :
             ((i==0 && j<(ny-2))?  rbx*(g_in.u[I3D(nx,nxy,i,j+2,k)] - g_in.u[i3d]) :
             ((i>1  && j==(ny-1))? -rby*(g_in.u[i3d] - g_in.u[I3D(nx,nxy,i-2,j,k)]) : 0.0)) ;

  REAL b_NE = (i<(nx-1)  && j<(ny-1))?  0.0 :
             ((i==(nx-1) && j<(ny-2))?  -rbx*(g_in.u[I3D(nx,nxy,i,j+2,k)] - g_in.u[i3d]) :
             ((i<(nx-2)  && j==(ny-1))? -rby*(g_in.u[I3D(nx,nxy,i+2,j,k)] - g_in.u[i3d]) : 0.0)) ;

  du2dt += ( r.xy[k]*( (g_in.u[SWxy] + b_SW) +
                       (g_in.u[NExy] + b_NE) -
                       (g_in.u[SExy] + b_SE) -
                       (g_in.u[NWxy] + b_NW) ) );

  #else

  /*------------------------------------------------------------------------
  * Isotropic Laplacian
  *------------------------------------------------------------------------
  */

  REAL du2dt = (  rCxyz_d*u
            +     rwe_d*(g_in.u[W] + g_in.u[E])
            +     rsn_d*(g_in.u[N] + g_in.u[S])
            +     rbt_d*(g_in.u[T] + g_in.u[B])
            +            rxyc_d*( g_in.u[SWxy] +
                                  g_in.u[SExy] +
                                  g_in.u[NWxy] +
                                  g_in.u[NExy] )
            +            rxzc_d*( g_in.u[SWxz] +
                                  g_in.u[SExz] +
                                  g_in.u[NWxz] +
                                  g_in.u[NExz] )
            +            ryzc_d*( g_in.u[SWyz] +
                                  g_in.u[SEyz] +
                                  g_in.u[NWyz] +
                                  g_in.u[NEyz] ) ) ;

  du2dt -= ( dt_d*( 0.5*I_sum
        +   rxyzf_d * ( ( J_d[E] + J_d[W] )
        +               ( J_d[N] + J_d[S] )
        +               ( J_d[B] + J_d[T] ) ) ) / C_m ) ;


  /*
  REAL du2dt = (
    +    ( g_in.u[W] - 2.f*u + g_in.u[E] )*rx_d
    +    ( g_in.u[N] - 2.f*u + g_in.u[S] )*ry_d
    +    ( g_in.u[T] - 2.f*u + g_in.u[B] )*rz_d );

  du2dt -= dt_d*I_sum/C_m ;
  */

  #endif

  /*------------------------------------------------------------------------
  * Time integration
  *------------------------------------------------------------------------
  */

  u += du2dt ;
  g_out.u[i3d] = u ;

  /*------------------------------------------------------------------------
  * Single precision
  *------------------------------------------------------------------------
  */

  #else

  REAL u = g_in.u[i3d] ;
  REAL v = g_in.v[i3d] ;
  REAL w = g_in.w[i3d] ;

  /*------------------------------------------------------------------------
  * Additional heaviside functions
  *------------------------------------------------------------------------
  */

  REAL  p     = ( u > theta_c   )   ? 1.0:0.0 ;
  REAL  q     = ( u > theta_v   )   ? 1.0:0.0 ;

  /*------------------------------------------------------------------------
  * Calculating dependant tau's
  *------------------------------------------------------------------------
  */

  REAL tau_vnu  =   q*tau_v1n + (1.f-q)*tau_v2n;

  g_out.v[i3d] =
    ( u > theta_c  ) ? v*expTau_vp_d : 1.0f-(1.0f-v)*expf(-dt_d/tau_vnu);

  g_out.w[i3d] =
    ( u > theta_c  ) ? w*expTau_wp_d : 1.0f-(1.0f-w)*expTau_wn_d;

/*
  REAL  dv2dt   =   (1.f-p)*(1.f-v)/tau_vnu - p*v/tau_vp;
  v += dv2dt*dt_d ;
  g_out.v[i3d] = v ;

  REAL dw2dt    =   (1.f-p)*(1.f-w)/tau_wn - p*w/tau_wp;
  w += dw2dt*dt_d ;
  g_out.w[i3d] = w ;
*/

  /*------------------------------------------------------------------------
  * I_sum
  *------------------------------------------------------------------------
  */

  //Fast inward  (Sodium)
  REAL  J_fi    =    -p*v*(1.f-u)*(u-theta_c)/tau_d;

  //Slow outward (Potassium)
  REAL  J_so    =    (1.f-p)*u/tau_o + p/tau_r;

  //Slow inward  (Calcium)
  REAL  J_si    =    -w*(1.f + tanhf(K*(u-u_csi)))/(2.f*tau_si);


  REAL  I_sum   =    J_fi + J_so + J_si ;

  J_d[i3d] = I_sum;

  /*------------------------------------------------------------------------
  * Laplacian Calculation
  *
  * No flux boundary condition is applied on all boundaries through
  * the Laplacian operator definition
  *------------------------------------------------------------------------
  */

  int S = ( j> 0      ) ? I3D(nx,nxy,i,j-1,k) : I3D(nx,nxy,i,j+1,k) ;
  int N = ( j<(ny-1)  ) ? I3D(nx,nxy,i,j+1,k) : I3D(nx,nxy,i,j-1,k) ;
  int W = ( i> 0      ) ? I3D(nx,nxy,i-1,j,k) : I3D(nx,nxy,i+1,j,k) ;
  int E = ( i<(nx-1)  ) ? I3D(nx,nxy,i+1,j,k) : I3D(nx,nxy,i-1,j,k) ;

  int SWxy = (i>0  && j>0) ? I3D(nx,nxy,i-1,j-1,k) :
            ((i==0 && j>0) ? I3D(nx,nxy,i+1,j-1,k) :
            ((i>0  && j==0)? I3D(nx,nxy,i-1,j+1,k) : I3D(nx,nxy,i+1,j+1,k) ) ) ;

  int SExy = (i<(nx-1)  && j>0) ? I3D(nx,nxy,i+1,j-1,k) :
            ((i==(nx-1) && j>0) ? I3D(nx,nxy,i-1,j-1,k) :
            ((i<(nx-1)  && j==0)? I3D(nx,nxy,i+1,j+1,k) : I3D(nx,nxy,i-1,j+1,k) ) ) ;

  int NWxy = (i>0  && j<(ny-1)) ? I3D(nx,nxy,i-1,j+1,k) :
            ((i==0 && j<(ny-1)) ? I3D(nx,nxy,i+1,j+1,k) :
            ((i>0  && j==(ny-1))? I3D(nx,nxy,i-1,j-1,k) : I3D(nx,nxy,i+1,j-1,k) ) ) ;

  int NExy = (i<(nx-1)  && j<(ny-1)) ? I3D(nx,nxy,i+1,j+1,k) :
            ((i==(nx-1) && j<(ny-1)) ? I3D(nx,nxy,i-1,j+1,k) :
            ((i<(nx-1)  && j==(ny-1))? I3D(nx,nxy,i+1,j-1,k) : I3D(nx,nxy,i-1,j-1,k) ) ) ;


  #ifdef PERIODIC_Z // In the z direction

  int B = ( k> 0      ) ? I3D(nx,nxy,i,j,k-1) : I3D(nx,nxy,i,j,nz-1) ;
  int T = ( k<(nz-1)  ) ? I3D(nx,nxy,i,j,k+1) : I3D(nx,nxy,i,j,0) ;            

  int SWxz = (i>0  && k>0) ? I3D(nx,nxy,i-1,j,k-1) :
          ((i==0 && k>0) ? I3D(nx,nxy,i+1,j,k-1) :
          ((i>0  && k==0)? I3D(nx,nxy,i-1,j,nz-1) : I3D(nx,nxy,i+1,j,k+1) ) ) ;

  int SExz = (i<(nx-1)  && k>0) ? I3D(nx,nxy,i+1,j,k-1) :
            ((i==(nx-1) && k>0) ? I3D(nx,nxy,i-1,j,k-1) :
            ((i<(nx-1)  && k==0)? I3D(nx,nxy,i+1,j,nz-1) : I3D(nx,nxy,i-1,j,k+1) ) ) ;

  int NWxz = (i>0  && k<(nz-1)) ? I3D(nx,nxy,i-1,j,k+1) :
            ((i==0 && k<(nz-1)) ? I3D(nx,nxy,i+1,j,k+1) :
            ((i>0  && k==(nz-1))? I3D(nx,nxy,i-1,j,0) : I3D(nx,nxy,i+1,j,k-1) ) ) ;

  int NExz = (i<(nx-1)  && k<(nz-1)) ? I3D(nx,nxy,i+1,j,k+1) :
            ((i==(nx-1) && k<(nz-1)) ? I3D(nx,nxy,i-1,j,k+1) :
            ((i<(nx-1)  && k==(nz-1))? I3D(nx,nxy,i+1,j,0) : I3D(nx,nxy,i-1,j,k-1) ) ) ;

//////////////////////////

  int SWyz = (j>0  && k>0) ? I3D(nx,nxy,i,j-1,k-1) :
            ((j==0 && k>0) ? I3D(nx,nxy,i,j+1,k-1) :
            ((j>0  && k==0)? I3D(nx,nxy,i,j-1,nz-1) : I3D(nx,nxy,i,j+1,k+1) ) ) ;

  int SEyz = (j<(ny-1)  && k>0) ? I3D(nx,nxy,i,j+1,k-1) :
            ((j==(ny-1) && k>0) ? I3D(nx,nxy,i,j-1,k-1) :
            ((j<(ny-1)  && k==0)? I3D(nx,nxy,i,j+1,nz-1) : I3D(nx,nxy,i,j-1,k+1) ) ) ;

  int NWyz = (j>0  && k<(nz-1)) ? I3D(nx,nxy,i,j-1,k+1) :
            ((j==0 && k<(nz-1)) ? I3D(nx,nxy,i,j+1,k+1) :
            ((j>0  && k==(nz-1))? I3D(nx,nxy,i,j-1,0) : I3D(nx,nxy,i,j+1,k-1) ) ) ;

  int NEyz = (j<(ny-1)  && k<(nz-1)) ? I3D(nx,nxy,i,j+1,k+1) :
            ((j==(ny-1) && k<(nz-1)) ? I3D(nx,nxy,i,j-1,k+1) :
            ((j<(ny-1)  && k==(nz-1))? I3D(nx,nxy,i,j+1,0) : I3D(nx,nxy,i,j-1,k-1) ) ) ;

  #else

  int B = ( k> 0      ) ? I3D(nx,nxy,i,j,k-1) : I3D(nx,nxy,i,j,k+1) ;
  int T = ( k<(nz-1)  ) ? I3D(nx,nxy,i,j,k+1) : I3D(nx,nxy,i,j,k-1) ;

  int SWxz = (i>0  && k>0) ? I3D(nx,nxy,i-1,j,k-1) :
            ((i==0 && k>0) ? I3D(nx,nxy,i+1,j,k-1) :
            ((i>0  && k==0)? I3D(nx,nxy,i-1,j,k+1) : I3D(nx,nxy,i+1,j,k+1) ) ) ;

  int SExz = (i<(nx-1)  && k>0) ? I3D(nx,nxy,i+1,j,k-1) :
            ((i==(nx-1) && k>0) ? I3D(nx,nxy,i-1,j,k-1) :
            ((i<(nx-1)  && k==0)? I3D(nx,nxy,i+1,j,k+1) : I3D(nx,nxy,i-1,j,k+1) ) ) ;

  int NWxz = (i>0  && k<(nz-1)) ? I3D(nx,nxy,i-1,j,k+1) :
            ((i==0 && k<(nz-1)) ? I3D(nx,nxy,i+1,j,k+1) :
            ((i>0  && k==(nz-1))? I3D(nx,nxy,i-1,j,k-1) : I3D(nx,nxy,i+1,j,k-1) ) ) ;

  int NExz = (i<(nx-1)  && k<(nz-1)) ? I3D(nx,nxy,i+1,j,k+1) :
            ((i==(nx-1) && k<(nz-1)) ? I3D(nx,nxy,i-1,j,k+1) :
            ((i<(nx-1)  && k==(nz-1))? I3D(nx,nxy,i+1,j,k-1) : I3D(nx,nxy,i-1,j,k-1) ) ) ;

//////////////////////////

  int SWyz = (j>0  && k>0) ? I3D(nx,nxy,i,j-1,k-1) :
            ((j==0 && k>0) ? I3D(nx,nxy,i,j+1,k-1) :
            ((j>0  && k==0)? I3D(nx,nxy,i,j-1,k+1) : I3D(nx,nxy,i,j+1,k+1) ) ) ;

  int SEyz = (j<(ny-1)  && k>0) ? I3D(nx,nxy,i,j+1,k-1) :
            ((j==(ny-1) && k>0) ? I3D(nx,nxy,i,j-1,k-1) :
            ((j<(ny-1)  && k==0)? I3D(nx,nxy,i,j+1,k+1) : I3D(nx,nxy,i,j-1,k+1) ) ) ;

  int NWyz = (j>0  && k<(nz-1)) ? I3D(nx,nxy,i,j-1,k+1) :
            ((j==0 && k<(nz-1)) ? I3D(nx,nxy,i,j+1,k+1) :
            ((j>0  && k==(nz-1))? I3D(nx,nxy,i,j-1,k-1) : I3D(nx,nxy,i,j+1,k-1) ) ) ;

  int NEyz = (j<(ny-1)  && k<(nz-1)) ? I3D(nx,nxy,i,j+1,k+1) :
            ((j==(ny-1) && k<(nz-1)) ? I3D(nx,nxy,i,j-1,k+1) :
            ((j<(ny-1)  && k==(nz-1))? I3D(nx,nxy,i,j+1,k-1) : I3D(nx,nxy,i,j-1,k-1) ) ) ;


 #endif

 #ifdef ANISOTROPIC_TISSUE

  /*------------------------------------------------------------------------
  * Anisotropic Laplacian
  *-------------------------------------------------------------------------
  */

  REAL rx = r.x[k];
  REAL ry = r.y[k];
  REAL rz = r.z[k];
  REAL rbx = r.bx[k];
  REAL rby = r.by[k];

  REAL du2dt = (  rCxyz_d * (rx + ry + rz)*u
            +    rwe_d * (4.0*rx - ry - rz)*(g_in.u[W] + g_in.u[E])
            +    rsn_d * (4.0*ry - rx - rz)*(g_in.u[N] + g_in.u[S])
            +    rbt_d * (4.0*rz - ry - rx)*(g_in.u[T] + g_in.u[B])
            +    rxyc_d * (rx + ry)*(  g_in.u[SWxy] +
                                       g_in.u[SExy] +
                                       g_in.u[NWxy] +
                                       g_in.u[NExy] )
            +    rxzc_d * (rx + rz)*(  g_in.u[SWxz] +
                                       g_in.u[SExz] +
                                       g_in.u[NWxz] +
                                       g_in.u[NExz] )
            +    ryzc_d * (ry + rz)*(  g_in.u[SWyz] +
                                       g_in.u[SEyz] +
                                       g_in.u[NWyz] +
                                       g_in.u[NEyz] ) ) ;

  du2dt -= ( dt_d*( 0.5*I_sum
        +   rxyzf_d * ( ( J_d[E] + J_d[W] )
        +               ( J_d[N] + J_d[S] )
        +               ( J_d[B] + J_d[T] ) ) ) / C_m ) ;

/*
  REAL du2dt = (
    +    ( g_in.u[W] - 2.f*u + g_in.u[E] )*rx
    +    ( g_in.u[N] - 2.f*u + g_in.u[S] )*ry
    +    ( g_in.u[T] - 2.f*u + g_in.u[B] )*rz );

  du2dt -= dt_d*I_sum/C_m ;
*/

  // Correction to NSWE boundary conditions
  REAL b_S = (j > 0 ) ? 0.f :
            ((j==0 && (i==0 || i==(nx-1)))? 0.f:
            rby*(g_in.u[I3D(nx,nxy,i+1,j,k)] - g_in.u[I3D(nx,nxy,i-1,j,k)]));

  REAL b_N = (j < (ny-1)) ? 0.f :
            ((j==(ny-1) && (i==0 || i==(nx-1)))? 0.f:
            -rby*(g_in.u[I3D(nx,nxy,i+1,j,k)] - g_in.u[I3D(nx,nxy,i-1,j,k)]));

  REAL b_W = (i > 0 ) ? 0.f :
            ((i==0 && (j==0 || j==(ny-1)))? 0.f:
            rbx*(g_in.u[I3D(nx,nxy,i,j+1,k)] - g_in.u[I3D(nx,nxy,i,j-1,k)]));

  REAL b_E = (i < (nx-1)) ? 0.f :
            ((i==(nx-1) && (j==0 || j==(ny-1)))? 0.f:
            -rbx*(g_in.u[I3D(nx,nxy,i,j+1,k)] - g_in.u[I3D(nx,nxy,i,j-1,k)]));

  du2dt += (
           (  b_S + b_N )*ry
       +   (  b_W + b_E )*rx  );

  // Correcion to SW SE NW NE boundary conditions
  REAL b_SW = (i>0 && j>0) ? 0.0f :
  ((i==0 && j>1) ? rbx*(g_in.u[i3d] - g_in.u[I3D(nx,nxy,i,j-2,k)]) :
  ((i>1  && j==0) ? rby*(g_in.u[i3d] - g_in.u[I3D(nx,nxy,i-2,j,k)]) : 0.0f));

  REAL b_SE = (i<(nx-1) && j>0) ? 0.0f :
  ((i==(nx-1) && j>1) ? - rbx*(g_in.u[i3d] - g_in.u[I3D(nx,nxy,i,j-2,k)]) :
  ((i<(nx-2)  && j==0) ? rby*(g_in.u[I3D(nx,nxy,i+2,j,k)] - g_in.u[i3d]) : 0.0f));

  REAL b_NW = (i>0 && j<(ny-1)) ? 0.0f :
  ((i==0 && j<(ny-2)) ? rbx*(g_in.u[I3D(nx,nxy,i,j+2,k)] - g_in.u[i3d]) :
  ((i>1  && j==(ny-1)) ? - rby*(g_in.u[i3d] - g_in.u[I3D(nx,nxy,i-2,j,k)]) : 0.0f));

  REAL b_NE = (i<(nx-1) && j<(ny-1)) ? 0.0f :
  ((i==(nx-1) && j<(ny-2)) ? - rbx*(g_in.u[I3D(nx,nxy,i,j+2,k)] - g_in.u[i3d]) :
  ((i<(nx-2)  && j==(ny-1)) ? - rby*(g_in.u[I3D(nx,nxy,i+2,j,k)] - g_in.u[i3d]) : 0.0f));

  du2dt += ( r.xy[k]*( (g_in.u[SWxy] + b_SW) +
                       (g_in.u[NExy] + b_NE) -
                       (g_in.u[SExy] + b_SE) -
                       (g_in.u[NWxy] + b_NW) ) );

  #else

  /*------------------------------------------------------------------------
  * Isotropic Laplacian
  *------------------------------------------------------------------------
  */

  REAL du2dt = (  rCxyz_d*u
            +     rwe_d*(g_in.u[W] + g_in.u[E])
            +     rsn_d*(g_in.u[N] + g_in.u[S])
            +     rbt_d*(g_in.u[T] + g_in.u[B])
            +            rxyc_d*( g_in.u[SWxy] +
                                  g_in.u[SExy] +
                                  g_in.u[NWxy] +
                                  g_in.u[NExy] )
            +            rxzc_d*( g_in.u[SWxz] +
                                  g_in.u[SExz] +
                                  g_in.u[NWxz] +
                                  g_in.u[NExz] )
            +            ryzc_d*( g_in.u[SWyz] +
                                  g_in.u[SEyz] +
                                  g_in.u[NWyz] +
                                  g_in.u[NEyz] ) ) ;

  du2dt -= ( dt_d*( 0.5f*I_sum
        +   rxyzf_d * ( ( J_d[E] + J_d[W] )
        +               ( J_d[N] + J_d[S] )
        +               ( J_d[B] + J_d[T] ) ) ) / C_m ) ;

/*
  REAL du2dt = (
    +    ( g_in.u[W] - 2.f*u + g_in.u[E] )*rx_d
    +    ( g_in.u[N] - 2.f*u + g_in.u[S] )*ry_d
    +    ( g_in.u[T] - 2.f*u + g_in.u[B] )*rz_d );

  du2dt -= dt_d*I_sum/C_m ;
*/

  #endif

  /*------------------------------------------------------------------------
  * Time integration
  *------------------------------------------------------------------------
  */

  u += du2dt ;
  g_out.u[i3d] = u ;

  #endif

  }


void FK_3V_wrapper(dim3 grid3D, dim3 block3D, stateVar gOut_d, stateVar gIn_d,
   conductionVar r_d, REAL *J_current_d) {

/*
      cudaEvent_t start,stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      float elapsedTime;
      cudaEventRecord(start,0);
*/

    FK_3V_kernel<<<grid3D, block3D>>>(gOut_d, gIn_d, r_d, J_current_d);
    CudaCheckError();
/*
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("3V kernel took: %f  ms\n", elapsedTime);
  //t += 2.0*dt;
*/

 /*   
    SIM_2V_kernel<<<grid3D, block3D>>>(gIn_d, gOut_d, r_d, J_current_d);
    CudaCheckError();
*/
    //swapSoA(&gIn_d, &gOut_d);

}

// This function launches all functions that need to be prcessed at every frame
// No graphics functions are launched from here
void animation(dim3 grid3D, dim3 block3D,
  stateVar g_h, stateVar gOut_d, stateVar gIn_d, REAL *J_current_d,
  conductionVar r_d, paramVar param, REAL *pt_h, REAL *pt_d,
  std::vector<electrodeVar> &electrode,
  bool initConditionFlag) {

  #pragma unroll
  for (int i=0;i<(ITPERFRAME);i++) {
    FK_3V_wrapper(grid3D,block3D,gOut_d,gIn_d,r_d,J_current_d);

    swapSoA(&gIn_d, &gOut_d);
  }

  // Single point time tracking
  singlePoint(gIn_d,pt_h,pt_d,param.singlePointPixel,electrode);


}

__global__ void singlePoint_kernel(stateVar g_in, REAL *pt_d,
   int singlePointPixel) {

  pt_d[0] = g_in.u[singlePointPixel];
  pt_d[1] = g_in.v[singlePointPixel];
  pt_d[2] = g_in.v[singlePointPixel];

}

void singlePoint(stateVar gIn_d, REAL *pt_h, REAL *pt_d,
   int singlePointPixel, std::vector<electrodeVar> &electrode) {

  singlePoint_kernel<<<1,1>>>(gIn_d, pt_d, singlePointPixel);
  CudaCheckError();

  CudaSafeCall(cudaMemcpy(pt_h, pt_d, 3*sizeof(REAL),cudaMemcpyDeviceToHost));

  electrodeVar data = {
    pt_h[0],
    pt_h[1],
    pt_h[2],
  };

  electrode.push_back(data);

}

__global__ void copyRender_kernel(int totpoints, stateVar g_in,
  VolumeType *h_volume) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < totpoints; i += stride) {
    h_volume[i] = (unsigned char)255.f*(float)g_in.u[i]*0.9f;

	  }

}

// Convert numerical values of the PDE solution to colors (char)
//extern "C"
void copyRender(dim3 grid1D, dim3 block1D, int totpoints,
  stateVar gIn_d, VolumeType *h_volume) {

/*
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  cudaEventRecord(start,0);
*/

  copyRender_kernel<<<grid1D, block1D>>>(totpoints, gIn_d, h_volume);
  CudaCheckError();

/*
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  printf("Time: %f  ms\n", elapsedTime);
*/

}

__global__ void spiralTip_kernel(REAL *g_past, stateVar g_present, 
  VolumeType *h_vol) {

  /*------------------------------------------------------------------------
  * Getting i, and k global indices
  *------------------------------------------------------------------------
  */

  const int i  = blockIdx.x*blockDim.x + threadIdx.x;
  const int j  = blockIdx.y;
  const int k  = threadIdx.y;

  /*------------------------------------------------------------------------
  * Return if we are outside the domain
  *------------------------------------------------------------------------
  */

  if( i >= nx && j>=ny && k >= nz) {
      return ;
  }

  const int nxy = nx*ny;

  int s0   = I3D(nx,nxy,i,j,k);
  int sx   = ( i<(nx-1)  ) ? I3D(nx,nxy,i+1,j,k) : I3D(nx,nxy,i,j,k);
  int sy   = ( j<(ny-1)  ) ? I3D(nx,nxy,i,j+1,k) : I3D(nx,nxy,i,j,k);
  int sz   = ( k<(nz-1)  ) ? I3D(nx,nxy,i,j,k+1) : I3D(nx,nxy,i,j,k);
  int sxy  = ( (j<(ny-1)) && (i<(nx-1) ) ) ? I3D(nx,nxy,i+1,j+1,k) : I3D(nx,nxy,i,j,k);
  int sxz  = ( (k<(nz-1)) && (i<(nx-1) ) ) ? I3D(nx,nxy,i+1,j,k+1) : I3D(nx,nxy,i,j,k);
  int syz  = ( (j<(ny-1)) && (k<(nz-1) ) ) ? I3D(nx,nxy,i,j+1,k+1) : I3D(nx,nxy,i,j,k);

  #ifdef SPIRALTIP_INTERPOLATION

  /*------------------------------------------------------------------------
  * Calculate pixel position of filament
  *------------------------------------------------------------------------
  */

  REAL x1, x2, x3, x4, y1, y2, y3, y4;
  REAL x3y1, x4y1, x3y2, x4y2, x1y3, x2y3, x1y4, x2y4, x2y1, x1y2, x4y3, x3y4;
  REAL den1, den2, ctn1, ctn2, disc, xtip, ytip, px, py, sroot1, sroot2;
  REAL gx, gy, gz, gx1, gx2, gx3, gx4, gy1, gy2, gy3, gy4, gz1, gz2, gz3, gz4;

  /*------------------------------------------------------------------------
  * Calculate pixel position of filament and plot
  *------------------------------------------------------------------------
  */


  int S = ( j>0 )                  ? I3D(nx,nxy,i,j-1,k) : I3D(nx,nxy,i,j+1,k) ;
  int Sx = ( (j>0) && (i<(nx-1)) ) ? I3D(nx,nxy,i+1,j-1,k) : I3D(nx,nxy,i-1,j+1,k) ;
  int Sy =                           I3D(nx,nxy,i,j,k) ;
  int Sxy = ( i<(nx-1) )           ? I3D(nx,nxy,i+1,j,k) : I3D(nx,nxy,i-1,j,k) ;
  int Sz = ( j>0 )  ? I3D(nx,nxy,i,j-1,k+1) : I3D(nx,nxy,i,j+2,k+1) ;
  int Sxz = ( j>0 ) ? I3D(nx,nxy,i+1,j-1,k+1) : I3D(nx,nxy,i,j+2,k+1) ;
  int Syz = ( j>0 ) ? I3D(nx,nxy,i,j,k+1) : I3D(nx,nxy,i,j+2,k+1) ;

  int N = ( j<(ny-1)  )                  ? I3D(nx,nxy,i,j+1,k) : I3D(nx,nxy,i,j-1,k) ;
  int Nx = ( (i<(nx-1)) && (j<(ny-1)) )  ? I3D(nx,nxy,i+1,j+1,k) : I3D(nx,nxy,i-1,j-1,k) ;
  int Ny = ( j<(ny-2) )                  ? I3D(nx,nxy,i,j+2,k) : (( j==(ny-2) ) ? I3D(nx,nxy,i,j,k) : I3D(nx,nxy,i,j-1,k)) ;
  int Nxy = ( (i<(nx-1)) && (j<(ny-2)) ) ? I3D(nx,nxy,i+1,j+2,k) : ((j==(ny-2)) ?  I3D(nx,nxy,i-1,j,k) : I3D(nx,nxy,i-1,j-1,k)) ;
  int Nz = ( (j<(ny-1)) && (k<(nz-1)) )                ? I3D(nx,nxy,i,j+1,k+1) : I3D(nx,nxy,i,j-1,k-1) ;
  int Nxz = ( (i<(nx-1)) && (j<(ny-1)) && (k<(nz-1)) ) ? I3D(nx,nxy,i+1,j+1,k+1) : I3D(nx,nxy,i-1,j-1,k-1) ;
  int Nyz = ( (j<(ny-2)) && (k<(nz-1)) )               ? I3D(nx,nxy,i,j+2,k+1) : ((k==(nz-2)) ? I3D(nx,nxy,i,j,k-1) : I3D(nx,nxy,i,j-1,k-1) );

  int W = ( i>0 )                  ? I3D(nx,nxy,i-1,j,k) : I3D(nx,nxy,i-1,j,k) ;
  int Wx =                           I3D(nx,nxy,i,j,k) ;
  int Wy = ( (i>0) && (j<(ny-1)) ) ? I3D(nx,nxy,i-1,j+1,k) : I3D(nx,nxy,i+1,j-1,k) ;
  int Wxy = ( (j<(ny-1)) )         ? I3D(nx,nxy,i,j+1,k) : I3D(nx,nxy,i-1,j,k) ;
  int Wz = ( (i>0) && (k<(nz-1)) )                ? I3D(nx,nxy,i-1,j,k+1) : I3D(nx,nxy,i+1,j,k-1) ;
  int Wxz = ( k<(nz-1) )                          ? I3D(nx,nxy,i,j,k+1) : I3D(nx,nxy,i,j,k-1) ;
  int Wyz = ( (i>0) && (j<(ny-1)) && (k<(nz-1)) ) ? I3D(nx,nxy,i-1,j+1,k+1) : I3D(nx,nxy,i+1,j-1,k-1) ;

  int E = ( i<(nx-1)  )                  ? I3D(nx,nxy,i+1,j,k) : I3D(nx,nxy,i-1,j,k) ;
  int Ex = ( i<(nx-2) )                  ? I3D(nx,nxy,i+2,j,k) : ((i==(nx-2)) ? I3D(nx,nxy,i,j,k) : I3D(nx,nxy,i-1,j,k));
  int Ey = ( (i<(nx-1)) && (j<(ny-1)) )  ? I3D(nx,nxy,i+1,j+1,k) : I3D(nx,nxy,i-1,j-1,k) ;
  int Exy = ( (i<(nx-2)) && (j<(ny-1)) ) ? I3D(nx,nxy,i+2,j+1,k) : ( (i==(nx-2)) ? I3D(nx,nxy,i,j-1,k) : I3D(nx,nxy,i-1,j-1,k)) ;
  int Ez = ( (i<(nx-1)) && (k<(nz-1)) )                ? I3D(nx,nxy,i+1,j,k+1) : I3D(nx,nxy,i-1,j,k-1) ;
  int Exz = ( (i<(nx-2)) && (k<(nz-1)) )               ? I3D(nx,nxy,i+2,j,k+1) : ( (i==(nx-2)) ? I3D(nx,nxy,i,j,k-1) : I3D(nx,nxy,i-1,j,k-1) );
  int Eyz = ( (i<(nx-1)) && (j<(ny-1)) && (k<(nz-1)) ) ? I3D(nx,nxy,i+1,j+1,k+1) : I3D(nx,nxy,i-1,j-1,k-1) ;

  int B = ( k>0 )                                 ? I3D(nx,nxy,i,j,k-1) : I3D(nx,nxy,i,j,k+1) ;
  int Bx = ( (k>0) && (i<(nx-1)) )                ? I3D(nx,nxy,i+1,j,k-1) : I3D(nx,nxy,i-1,j,k+1) ;
  int By = ( (k>0) && (j<(ny-1)) )                ? I3D(nx,nxy,i,j+1,k-1) : I3D(nx,nxy,i,j-1,k+1) ;
  int Bxy = ( (i<(nx-1)) && (j<(ny-1)) && (k>0) ) ? I3D(nx,nxy,i+1,j+1,k-1) : I3D(nx,nxy,i-1,j-1,k+1) ;
  int Bz =                 I3D(nx,nxy,i,j,k);
  int Bxz = ( i<(nx-1) ) ? I3D(nx,nxy,i+1,j,k) : I3D(nx,nxy,i-1,j,k) ;
  int Byz = ( j<(ny-1) ) ? I3D(nx,nxy,i,j+1,k) : I3D(nx,nxy,i,j-1,k) ;

  int T = ( k<(nz-1)  )                                ? I3D(nx,nxy,i,j,k+1) : I3D(nx,nxy,i,j,k-1) ;
  int Tx = ( (i<(nx-1)) && (k<(nz-1)) )                ? I3D(nx,nxy,i+1,j,k+1) : I3D(nx,nxy,i-1,j,k-1) ;
  int Ty = ( (j<(ny-1)) && (k<(nz-1)) )                ? I3D(nx,nxy,i,j+1,k+1) : I3D(nx,nxy,i,j-1,k-1) ;
  int Txy = ( (i<(nx-1)) && (j<(ny-1)) && (k<(nz-1)) ) ? I3D(nx,nxy,i+1,j+1,k+1) : I3D(nx,nxy,i-1,j-1,k-1) ;
  int Tz = ( k<(nz-2)  )                 ? I3D(nx,nxy,i,j,k+2) : ( (k==(nz-2)) ? I3D(nx,nxy,i,j,k) : I3D(nx,nxy,i,j,k-1));
  int Txz = ( (i<(nx-1)) && k<(nz-2) )   ? I3D(nx,nxy,i+1,j,k+2) : ( (k==(nz-2)) ? I3D(nx,nxy,i-1,j,k) : I3D(nx,nxy,i-1,j,k-1));
  int Tyz = ( (j<(ny-1)) && (k<(nz-2)) ) ? I3D(nx,nxy,i,j+1,k+2) : ( (k==(nz-2)) ? I3D(nx,nxy,i,j-1,k) : I3D(nx,nxy,i,j-1,k-1) );

  /*------------------------------------------------------------------------
  * XY plane
  *------------------------------------------------------------------------
  */

  x1 = g_present.u[s0];
  x2 = g_present.u[sx];
  x4 = g_present.u[sy];
  x3 = g_present.u[sxy];

  y1 = g_past[s0];
  y2 = g_past[sx];
  y4 = g_past[sy];
  y3 = g_past[sxy];

  x3y1 = x3*y1;
  x4y1 = x4*y1;
  x3y2 = x3*y2;
  x4y2 = x4*y2;
  x1y3 = x1*y3;
  x2y3 = x2*y3;
  x1y4 = x1*y4;
  x2y4 = x2*y4;
  x2y1 = x2*y1;
  x1y2 = x1*y2;
  x4y3 = x4*y3;
  x3y4 = x3*y4;

  den1 = 2.0*(x3y1 - x4y1 - x3y2 + x4y2 - x1y3 + x2y3 + x1y4 - x2y4);
  den2 = 2.0*(x2y1 - x3y1 - x1y2 + x4y2 + x1y3 - x4y3 - x2y4 + x3y4);

  ctn1 = x1 - x2 + x3 - x4 - y1 + y2 - y3 + y4;
  ctn2 = x3y1 - 2.0*x4y1 + x4y2 - x1y3 + 2.0*x1y4 - x2y4;

  disc = 4.0 * ( x3y1 - x3y2 - x4y1 + x4y2 - x1y3 + x1y4 + x2y3 - x2y4 )
    * (x4y1 - x1y4 + Uth * (x1 - x4 - y1 + y4)) +
    ( -ctn2 + Uth * ctn1 ) * (-ctn2 + Uth * ctn1 );

  px = -(Uth * ctn1 - ctn2)/den1;
  py = (Uth * ctn1)/den2 -
    (-2.0* x2y1 + x3y1 + 2.0 *x1y2 - x4y2 - x1y3 + x2y4)/den2;

  sroot1 = sqrt(disc)/den1;
  sroot2 = sqrt(disc)/den2;

  /*------------------------------------------------------------------------
  * XY plane
  * Clockwise direction
  *------------------------------------------------------------------------
  */

  xtip = px + sroot1;
  ytip = py + sroot2;

  if ( ( ((xtip > 0.0) && (xtip < 1.0)) +
       ((ytip > 0.0) && (ytip < 1.0)) +
       ( disc > 0.0 ) ) > 2 ) {
/*
    gx = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz = (g_present.u[T] - g_present.u[B])*invdz_d;
*/

    gx1 = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy1 = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz1 = (g_present.u[T] - g_present.u[B])*invdz_d;

    gx2 = (g_present.u[Ex] - g_present.u[Wx])*invdx_d;
    gy2 = (g_present.u[Nx] - g_present.u[Sx])*invdy_d;
    gz2 = (g_present.u[Tx] - g_present.u[Bx])*invdz_d;

    gx3 = (g_present.u[Ey] - g_present.u[Wy])*invdx_d;
    gy3 = (g_present.u[Ny] - g_present.u[Sy])*invdy_d;
    gz3 = (g_present.u[Ty] - g_present.u[By])*invdz_d;

    gx4 = (g_present.u[Exy] - g_present.u[Wxy])*invdx_d;
    gy4 = (g_present.u[Nxy] - g_present.u[Sxy])*invdy_d;
    gz4 = (g_present.u[Txy] - g_present.u[Bxy])*invdz_d;

    gx = (1.0  - xtip)*(1.0 - ytip)*gx1 +
      xtip*(1.0 - ytip)*gx2 + ytip*(1.0 - xtip)*gx3 + xtip*ytip*gx4;

    gy = (1.0  - xtip)*(1.0 - ytip)*gy1 +
      xtip*(1.0 - ytip)*gy2 + ytip*(1.0 - xtip)*gy3 + xtip*ytip*gy4;

    gz = (1.0  - xtip)*(1.0 - ytip)*gz1 +
      xtip*(1.0 - ytip)*gz2 + ytip*(1.0 - xtip)*gz3 + xtip*ytip*gz4;

    vec3dyn a = { .x = (REAL)i+xtip, .y = (REAL)j+ytip, .z = (REAL)k};
    vec6dyn b = { .x = (REAL)i+xtip, .y = (REAL)j+ytip, .z = (REAL)k, .vx = gx, .vy = gy, .vz = gz};
    tip_push_back1(a);
    tip_push_back2(b);

    h_vol[I3D(nx,nxy,i,j,k)] = (unsigned char)255;
  }

  /*------------------------------------------------------------------------
  * Anticlockwise direction
  *------------------------------------------------------------------------
  */

  xtip = px - sroot1;
  ytip = py - sroot2;

  if ( ( ((xtip > 0.0) && (xtip < 1.0)) +
       ((ytip > 0.0) && (ytip < 1.0)) +
       ( disc > 0.0 ) ) > 2 ) {

/*
    gx = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz = (g_present.u[T] - g_present.u[B])*invdz_d;
*/

    gx1 = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy1 = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz1 = (g_present.u[T] - g_present.u[B])*invdz_d;

    gx2 = (g_present.u[Ex] - g_present.u[Wx])*invdx_d;
    gy2 = (g_present.u[Nx] - g_present.u[Sx])*invdy_d;
    gz2 = (g_present.u[Tx] - g_present.u[Bx])*invdz_d;

    gx3 = (g_present.u[Ey] - g_present.u[Wy])*invdx_d;
    gy3 = (g_present.u[Ny] - g_present.u[Sy])*invdy_d;
    gz3 = (g_present.u[Ty] - g_present.u[By])*invdz_d;

    gx4 = (g_present.u[Exy] - g_present.u[Wxy])*invdx_d;
    gy4 = (g_present.u[Nxy] - g_present.u[Sxy])*invdy_d;
    gz4 = (g_present.u[Txy] - g_present.u[Bxy])*invdz_d;

    gx = (1.0  - xtip)*(1.0 - ytip)*gx1 +
      xtip*(1.0 - ytip)*gx2 + ytip*(1.0 - xtip)*gx3 + xtip*ytip*gx4;

    gy = (1.0  - xtip)*(1.0 - ytip)*gy1 +
      xtip*(1.0 - ytip)*gy2 + ytip*(1.0 - xtip)*gy3 + xtip*ytip*gy4;

    gz = (1.0  - xtip)*(1.0 - ytip)*gz1 +
      xtip*(1.0 - ytip)*gz2 + ytip*(1.0 - xtip)*gz3 + xtip*ytip*gz4;

    vec3dyn a = { .x = (REAL)i+xtip, .y = (REAL)j+ytip, .z = (REAL)k};
    vec6dyn b = { .x = (REAL)i+xtip, .y = (REAL)j+ytip, .z = (REAL)k, .vx = gx, .vy = gy, .vz = gz};

    tip_push_back1(a);
    tip_push_back2(b);

    h_vol[I3D(nx,nxy,i,j,k)] = (unsigned char)255;
  }


  /*------------------------------------------------------------------------
  * XZ plane
  * Clockwise direction
  *------------------------------------------------------------------------
  */

  x1 = g_present.u[s0];
  x2 = g_present.u[sx];
  x3 = g_present.u[sxz];
  x4 = g_present.u[sz];

  y1 = g_past[s0];
  y2 = g_past[sx];
  y3 = g_past[sxz];
  y4 = g_past[sz];

  x3y1 = x3*y1;
  x4y1 = x4*y1;
  x3y2 = x3*y2;
  x4y2 = x4*y2;
  x1y3 = x1*y3;
  x2y3 = x2*y3;
  x1y4 = x1*y4;
  x2y4 = x2*y4;
  x2y1 = x2*y1;
  x1y2 = x1*y2;
  x4y3 = x4*y3;
  x3y4 = x3*y4;

  den1 = 2.0*(x3y1 - x4y1 - x3y2 + x4y2 - x1y3 + x2y3 + x1y4 - x2y4);
  den2 = 2.0*(x2y1 - x3y1 - x1y2 + x4y2 + x1y3 - x4y3 - x2y4 + x3y4);

  ctn1 = x1 - x2 + x3 - x4 - y1 + y2 - y3 + y4;
  ctn2 = x3y1 - 2.0*x4y1 + x4y2 - x1y3 + 2.0*x1y4 - x2y4;

  disc = 4.0 * ( x3y1 - x3y2 - x4y1 + x4y2 - x1y3 + x1y4 + x2y3 - x2y4 )
    * (x4y1 - x1y4 + Uth * (x1 - x4 - y1 + y4)) +
    ( -ctn2 + Uth * ctn1 ) * (-ctn2 + Uth * ctn1 );

  px = -(Uth * ctn1 - ctn2)/den1;
  py = (Uth * ctn1)/den2 -
    (-2.0* x2y1 + x3y1 + 2.0 *x1y2 - x4y2 - x1y3 + x2y4)/den2;

  sroot1 = sqrt(disc)/den1;
  sroot2 = sqrt(disc)/den2;

  /*------------------------------------------------------------------------
  * XZ plane
  * Clockwise direction
  *------------------------------------------------------------------------
  */

  xtip = px + sroot1;
  ytip = py + sroot2;

  if ( ( ((xtip > 0.0) && (xtip < 1.0)) +
       ((ytip > 0.0) && (ytip < 1.0)) +
       ( disc > 0.0 ) ) > 2 ) {

 /*
    gx = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz = (g_present.u[T] - g_present.u[B])*invdz_d;
*/

    gx1 = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy1 = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz1 = (g_present.u[T] - g_present.u[B])*invdz_d;

    gx2 = (g_present.u[Ex] - g_present.u[Wx])*invdx_d;
    gy2 = (g_present.u[Nx] - g_present.u[Sx])*invdy_d;
    gz2 = (g_present.u[Tx] - g_present.u[Bx])*invdz_d;

    gx3 = (g_present.u[Ez] - g_present.u[Wz])*invdx_d;
    gy3 = (g_present.u[Nz] - g_present.u[Sz])*invdy_d;
    gz3 = (g_present.u[Tz] - g_present.u[Bz])*invdz_d;

    gx4 = (g_present.u[Exz] - g_present.u[Wxz])*invdx_d;
    gy4 = (g_present.u[Nxz] - g_present.u[Sxz])*invdy_d;
    gz4 = (g_present.u[Txz] - g_present.u[Bxz])*invdz_d;

    gx = (1.0  - xtip)*(1.0 - ytip)*gx1 +
      xtip*(1.0 - ytip)*gx2 + ytip*(1.0 - xtip)*gx3 + xtip*ytip*gx4;

    gy = (1.0  - xtip)*(1.0 - ytip)*gy1 +
      xtip*(1.0 - ytip)*gy2 + ytip*(1.0 - xtip)*gy3 + xtip*ytip*gy4;

    gz = (1.0  - xtip)*(1.0 - ytip)*gz1 +
      xtip*(1.0 - ytip)*gz2 + ytip*(1.0 - xtip)*gz3 + xtip*ytip*gz4;

    vec3dyn a = { .x = (REAL)i+xtip, .y = (REAL)j, .z = (REAL)k+ytip};
    vec6dyn b = { .x = (REAL)i+xtip, .y = (REAL)j, .z = (REAL)k+ytip, .vx = gx, .vy = gy, .vz = gz};
    tip_push_back1(a);
    tip_push_back2(b);

    h_vol[I3D(nx,nxy,i,j,k)] = (unsigned char)255;

  }

  /*------------------------------------------------------------------------
  * Anticlockwise direction
  *------------------------------------------------------------------------
  */

  xtip = px - sroot1;
  ytip = py - sroot2;

  if ( ( ((xtip > 0.0) && (xtip < 1.0)) +
       ((ytip > 0.0) && (ytip < 1.0)) +
       ( disc > 0.0 ) ) > 2 ) {

 /*
    gx = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz = (g_present.u[T] - g_present.u[B])*invdz_d;
*/

    gx1 = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy1 = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz1 = (g_present.u[T] - g_present.u[B])*invdz_d;

    gx2 = (g_present.u[Ex] - g_present.u[Wx])*invdx_d;
    gy2 = (g_present.u[Nx] - g_present.u[Sx])*invdy_d;
    gz2 = (g_present.u[Tx] - g_present.u[Bx])*invdz_d;

    gx3 = (g_present.u[Ez] - g_present.u[Wz])*invdx_d;
    gy3 = (g_present.u[Nz] - g_present.u[Sz])*invdy_d;
    gz3 = (g_present.u[Tz] - g_present.u[Bz])*invdz_d;

    gx4 = (g_present.u[Exz] - g_present.u[Wxz])*invdx_d;
    gy4 = (g_present.u[Nxz] - g_present.u[Sxz])*invdy_d;
    gz4 = (g_present.u[Txz] - g_present.u[Bxz])*invdz_d;

    gx = (1.0  - xtip)*(1.0 - ytip)*gx1 +
      xtip*(1.0 - ytip)*gx2 + ytip*(1.0 - xtip)*gx3 + xtip*ytip*gx4;

    gy = (1.0  - xtip)*(1.0 - ytip)*gy1 +
      xtip*(1.0 - ytip)*gy2 + ytip*(1.0 - xtip)*gy3 + xtip*ytip*gy4;

    gz = (1.0  - xtip)*(1.0 - ytip)*gz1 +
      xtip*(1.0 - ytip)*gz2 + ytip*(1.0 - xtip)*gz3 + xtip*ytip*gz4;

    vec3dyn a = { .x = (REAL)i+xtip, .y = (REAL)j, .z = (REAL)k+ytip};
    vec6dyn b = { .x = (REAL)i+xtip, .y = (REAL)j, .z = (REAL)k+ytip, .vx = gx, .vy = gy, .vz = gz};

    tip_push_back1(a);
    tip_push_back2(b);

    h_vol[I3D(nx,nxy,i,j,k)] = (unsigned char)255;

  }


  /*------------------------------------------------------------------------
  * YZ direction
  * Anticlockwse direction
  *------------------------------------------------------------------------
  */

  x1 = g_present.u[s0];
  x2 = g_present.u[sy];
  x3 = g_present.u[syz];
  x4 = g_present.u[sz];

  y1 = g_past[s0];
  y2 = g_past[sy];
  y3 = g_past[syz];
  y4 = g_past[sz];

  x3y1 = x3*y1;
  x4y1 = x4*y1;
  x3y2 = x3*y2;
  x4y2 = x4*y2;
  x1y3 = x1*y3;
  x2y3 = x2*y3;
  x1y4 = x1*y4;
  x2y4 = x2*y4;
  x2y1 = x2*y1;
  x1y2 = x1*y2;
  x4y3 = x4*y3;
  x3y4 = x3*y4;

  den1 = 2.0*(x3y1 - x4y1 - x3y2 + x4y2 - x1y3 + x2y3 + x1y4 - x2y4);
  den2 = 2.0*(x2y1 - x3y1 - x1y2 + x4y2 + x1y3 - x4y3 - x2y4 + x3y4);

  ctn1 = x1 - x2 + x3 - x4 - y1 + y2 - y3 + y4;
  ctn2 = x3y1 - 2.0*x4y1 + x4y2 - x1y3 + 2.0*x1y4 - x2y4;

  disc = 4.0 * ( x3y1 - x3y2 - x4y1 + x4y2 - x1y3 + x1y4 + x2y3 - x2y4 )
    * (x4y1 - x1y4 + Uth * (x1 - x4 - y1 + y4)) +
    ( -ctn2 + Uth * ctn1 ) * (-ctn2 + Uth * ctn1 );

  px = -(Uth * ctn1 - ctn2)/den1;
  py = (Uth * ctn1)/den2 -
    (-2.0* x2y1 + x3y1 + 2.0 *x1y2 - x4y2 - x1y3 + x2y4)/den2;

  sroot1 = sqrt(disc)/den1;
  sroot2 = sqrt(disc)/den2;

  /*------------------------------------------------------------------------
  * YZ plane
  * Clockwise direction
  *------------------------------------------------------------------------
  */

  xtip = px + sroot1;
  ytip = py + sroot2;

  if ( ( ((xtip > 0.0) && (xtip < 1.0)) +
       ((ytip > 0.0) && (ytip < 1.0)) +
       ( disc > 0.0 ) ) > 2 ) {
 /*
    gx = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz = (g_present.u[T] - g_present.u[B])*invdz_d;
*/

    gx1 = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy1 = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz1 = (g_present.u[T] - g_present.u[B])*invdz_d;

    gx2 = (g_present.u[Ey] - g_present.u[Wy])*invdx_d;
    gy2 = (g_present.u[Ny] - g_present.u[Sy])*invdy_d;
    gz2 = (g_present.u[Ty] - g_present.u[By])*invdz_d;

    gx3 = (g_present.u[Ez] - g_present.u[Wz])*invdx_d;
    gy3 = (g_present.u[Nz] - g_present.u[Sz])*invdy_d;
    gz3 = (g_present.u[Tz] - g_present.u[Bz])*invdz_d;

    gx4 = (g_present.u[Eyz] - g_present.u[Wyz])*invdx_d;
    gy4 = (g_present.u[Nyz] - g_present.u[Syz])*invdy_d;
    gz4 = (g_present.u[Tyz] - g_present.u[Byz])*invdz_d;

    gx = (1.0  - xtip)*(1.0 - ytip)*gx1 +
      xtip*(1.0 - ytip)*gx2 + ytip*(1.0 - xtip)*gx3 + xtip*ytip*gx4;

    gy = (1.0  - xtip)*(1.0 - ytip)*gy1 +
      xtip*(1.0 - ytip)*gy2 + ytip*(1.0 - xtip)*gy3 + xtip*ytip*gy4;

    gz = (1.0  - xtip)*(1.0 - ytip)*gz1 +
      xtip*(1.0 - ytip)*gz2 + ytip*(1.0 - xtip)*gz3 + xtip*ytip*gz4;

    vec3dyn a = { .x = (REAL)i, .y = (REAL)j+xtip, .z = (REAL)k+ytip};
    vec6dyn b = { .x = (REAL)i, .y = (REAL)j+xtip, .z = (REAL)k+ytip, .vx = gx, .vy = gy, .vz = gz};
    tip_push_back1(a);
    tip_push_back2(b);

    h_vol[I3D(nx,nxy,i,j,k)] = (unsigned char)255;

  }

  /*------------------------------------------------------------------------
  * Anticlockwise direction
  *------------------------------------------------------------------------
  */

  xtip = px - sroot1;
  ytip = py - sroot2;

  if ( ( ((xtip > 0.0) && (xtip < 1.0)) +
       ((ytip > 0.0) && (ytip < 1.0)) +
       ( disc > 0.0 ) ) > 2 ) {

 /*
    gx = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz = (g_present.u[T] - g_present.u[B])*invdz_d;
*/

    gx1 = (g_present.u[E] - g_present.u[W])*invdx_d;
    gy1 = (g_present.u[N] - g_present.u[S])*invdy_d;
    gz1 = (g_present.u[T] - g_present.u[B])*invdz_d;

    gx2 = (g_present.u[Ey] - g_present.u[Wy])*invdx_d;
    gy2 = (g_present.u[Ny] - g_present.u[Sy])*invdy_d;
    gz2 = (g_present.u[Ty] - g_present.u[By])*invdz_d;

    gx3 = (g_present.u[Ez] - g_present.u[Wz])*invdx_d;
    gy3 = (g_present.u[Nz] - g_present.u[Sz])*invdy_d;
    gz3 = (g_present.u[Tz] - g_present.u[Bz])*invdz_d;

    gx4 = (g_present.u[Eyz] - g_present.u[Wyz])*invdx_d;
    gy4 = (g_present.u[Nyz] - g_present.u[Syz])*invdy_d;
    gz4 = (g_present.u[Tyz] - g_present.u[Byz])*invdz_d;

    gx = (1.0  - xtip)*(1.0 - ytip)*gx1 +
      xtip*(1.0 - ytip)*gx2 + ytip*(1.0 - xtip)*gx3 + xtip*ytip*gx4;

    gy = (1.0  - xtip)*(1.0 - ytip)*gy1 +
      xtip*(1.0 - ytip)*gy2 + ytip*(1.0 - xtip)*gy3 + xtip*ytip*gy4;

    gz = (1.0  - xtip)*(1.0 - ytip)*gz1 +
      xtip*(1.0 - ytip)*gz2 + ytip*(1.0 - xtip)*gz3 + xtip*ytip*gz4;

    vec3dyn a = { .x = (REAL)i, .y = (REAL)j+xtip, .z = (REAL)k+ytip};
    vec6dyn b = { .x = (REAL)i, .y = (REAL)j+xtip, .z = (REAL)k+ytip, .vx = gx, .vy = gy, .vz = gz};

    tip_push_back1(a);
    tip_push_back2(b);

    h_vol[I3D(nx,nxy,i,j,k)] = (unsigned char)255;

  }

  #else

  /*------------------------------------------------------------------------
  * Calculate tip for visualization
  *------------------------------------------------------------------------
  */
  
  int sxyz = ( (i<(nx-1)) && (j<(ny-1)) && (k<(nz-1) ) ) ? I3D(nx,nxy,i+1,j+1,k+1) : I3D(nx,nxy,i,j,k);

  if ( (i<(nx-1)) && (j<(ny-1)) && (k<(nz-1)) ) {
    h_vol[I3D(nx,nxy,i,j,k)] = 255*(unsigned char)(filament(s0,sx,sy,sz,sxy,sxz,syz,sxyz,g_past,g_present));
  }
  else {
    h_vol[I3D(nx,nxy,i,j,k)] = 0;
  }

  #endif

}


__device__ int tip_push_back1(vec3dyn & mt) {

  int insert_pt = atomicAdd(&dev_count, 1);
  if (insert_pt < NN){
    dev_data1[insert_pt] = mt;
    return insert_pt;}
  else return -1;

  }

__device__ int tip_push_back2(vec6dyn & mt) {

int insert_pt = dev_count;//atomicAdd(&dev_count, 1);
if (insert_pt < NN){
  dev_data2[insert_pt] = mt;
  return insert_pt;}
else return -1;

}


__device__ bool filament(int s0, int sx, int sy, int sz, int sxy, int sxz, int syz, int sxyz,
  REAL *g_past, stateVar g_present) {

  REAL v0, vx, vy, vz, vxy, vxz, vyz, vxyz;
  REAL d0, dx, dy, dz, dxy, dxz, dyz, dxyz;
  REAL f0, fx, fy, fz, fxy, fxz, fyz, fxyz;
  REAL s;
  
  bool bv, bdv;

  v0   = g_present.u[s0];
  vx   = g_present.u[sx];
  vy   = g_present.u[sy];
  vz   = g_present.u[sz];
  vxy  = g_present.u[sxy];
  vxz  = g_present.u[sxz];
  vyz  = g_present.u[syz];
  vxyz = g_present.u[sxyz];

  f0   = v0   - Uth;
  fx   = vx   - Uth;
  fy   = vy   - Uth;
  fz   = vz   - Uth;
  fxy  = vxy  - Uth;
  fyz  = vyz  - Uth;
  fxz  = vxz  - Uth;
  fxyz = vxyz - Uth;

  s = STEP(0.0, f0  )
    + STEP(0.0, fx  )
    + STEP(0.0, fy  )
    + STEP(0.0, fz  )
    + STEP(0.0, fxy )
    + STEP(0.0, fyz )
    + STEP(0.0, fxz )
    + STEP(0.0, fxyz);

  bv = ( s>0.5 ) && ( s<7.5 );

  d0   = v0   - g_past[s0];
  dx   = vx   - g_past[sx];
  dy   = vy   - g_past[sy];
  dz   = vz   - g_past[sz];
  dxy  = vxy  - g_past[sxy];
  dxz  = vxz  - g_past[sxz];
  dyz  = vyz  - g_past[syz];
  dxyz = vxyz - g_past[sxyz];

  s = STEP(0.0, d0  )
    + STEP(0.0, dx  )
    + STEP(0.0, dy  )
    + STEP(0.0, dz  )
    + STEP(0.0, dxy )
    + STEP(0.0, dyz )
    + STEP(0.0, dxz )
    + STEP(0.0, dxyz);

  bdv = ( s>0.5 ) && ( s<7.5 );

  return ( bdv && bv );

}


// Spiral tip tracking (not precise)
VolumeType *spiralTip(dim3 grid3Dz, dim3 block3Dz, REAL *v_past_d,
  stateVar gIn_d, VolumeType *h_volume) {
/*
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  cudaEventRecord(start,0);
*/
  spiralTip_kernel<<<grid3Dz,block3Dz>>>(v_past_d, gIn_d, h_volume);
  CudaCheckError();
/*
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);
  printf("Time: %f  ms\n", elapsedTime);
*/

  return h_volume;

}


// Set voltage to zero on certain regions of the domain (condution block) to
// initialize a spiral wave
void cutVoltage(paramVar p, stateVar g_h, stateVar g_present_d) {

  int i, j, k, idx;

  CudaSafeCall(cudaMemcpy(g_h.u, g_present_d.u, p.memSize,
    cudaMemcpyDeviceToHost));

  if (p.counterclock) {

    for (k=0;k<nz;k++) {
    	for (j=0;j<ny;j++) {
        for (i=nx/2;i<nx;i++) {
    			idx = i + nx * (j + ny * k);
    			g_h.u[idx] = 0.0;
    			}
    		}
    	}

    }

  if (p.clock) {

    for (k=0;k<nz;k++) {
    	for (j=0;j<ny;j++) {
        for (i=0;i<nx/2;i++) {
    			idx = i + nx * (j + ny * k);
    			g_h.u[idx] = 0.0;
    			}
    		}
    	}

  	}

  CudaSafeCall(cudaMemcpy(g_present_d.u, g_h.u, p.memSize,
    cudaMemcpyHostToDevice));

}

// Stimulate with voltage certain regions of the domain
void stimulateV(int memSize, stateVar g_h, stateVar g_present_d) {

  int i, j, k, idx;

  CudaSafeCall(cudaMemcpy(g_h.u, g_present_d.u, memSize,
    cudaMemcpyDeviceToHost));

  for (k=(int)floor(0);k<(int)floor(nz);k++) {
    for (j=(int)floor(0);j<(int)floor(ny/8);j++) {
      for (i=(int)floor(0);i<(int)floor(nx);i++) {
        idx = i + nx*j + nx*ny*k;
        g_h.u[idx] = 1.0f;
      }
    }
  }

  CudaSafeCall(cudaMemcpy(g_present_d.u, g_h.u, memSize,
    cudaMemcpyHostToDevice));

}
