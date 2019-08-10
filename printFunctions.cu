
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

#include "./common/CudaSafeCall.h"

extern __device__ vec3dyn dev_data1[NN];
extern __device__ vec6dyn dev_data2[NN];
extern __device__ int dev_count;

// Print a 1D slice of data
void print1D(stateVar g_h, int count, const char strA[], int x) {

	int i, j, k, idx;

	char strC[32];

	char strB[32];
	sprintf(strB, "data1d_%d.dat", count);

	strncpy(strC,strA,x);
	strC[x] = '\0';
	strcat(strC,strB);
	strcat(strC,strA+x);

	k = nz/2;
	i = nx/2;

	//Print data
	FILE *fp1;
	fp1 = fopen(strC,"w+");

	for (j=1;j<(ny-1);j++) {
		idx = i + nx * (j + ny * k);
		fprintf(fp1, "%d\t %f\n", j, (float)g_h.u[idx]);
		}

	fclose (fp1);

	printf("1D data file %d created\n", count);

}

// Print a 2D slice of data
void print2D(stateVar g_h, int count, const char strA[], int x) {

	int i, j, k, idx;

	char strC[32];

	char strB[32];
	sprintf(strB, "data2d_%d.dat", count);

	strncpy(strC,strA,x);
	strC[x] = '\0';
	strcat(strC,strB);
	strcat(strC,strA+x);

	k = nz/2;
	//i = nx/2;

	//Print data
	FILE *fp1;
	fp1 = fopen(strC,"w+");

	for (j=1;j<(ny-1);j++) {
			for (i=1;i<(nx-1);i++) {
				idx = i + nx * (j + ny * k);
						fprintf(fp1, "%d\t %d\t %f\n", i, j, (float)g_h.u[idx]);
					}
					fprintf(fp1,"\n");
				}

	fclose (fp1);

	printf("2D data file %d created\n", count);

}

// Print a 3D slice of data
void print3D(stateVar g_h, int count, const char strA[], int x) {

	int i, j, k, idx;

	char strC[128];

	char strB[256];
	sprintf(strB, "/time3D/data3d_%d.dat", count);

	strncpy(strC,strA,x);
	strC[x] = '\0';
	strcat(strC,strB);
	strcat(strC,strA+x);

	//Print data
	FILE *fp1;
	fp1 = fopen(strC ,"w+");

	for (k=0;k<nz;k++) {
		for (j=0;j<ny;j++) {
			for (i=0;i<nx;i++) {
				idx = i + nx * (j + ny * k);
				fprintf(fp1, "%f\t%f\t%f\n", 
					(float)g_h.u[idx], (float)g_h.v[idx], (float)g_h.w[idx]);
				}
			}
		}

	fclose (fp1);

	printf("3D data file %d created\n", count);

}


// Voltage time tracing
void printVoltageInTime(std::vector<electrodeVar> &sol, const char strA[],
	int x, REAL dt, int count) {

	int i;

	char strC[128];
	const char *strB = "datatime.dat";

	strncpy(strC,strA,x);
	strC[x] = '\0';
	strcat(strC,strB);
	strcat(strC,strA+x);

	//Print data
	FILE *fp1;
	fp1 = fopen(strC,"w+");

	for (i=0;i<sol.size();i++) {
		fprintf(fp1, "%f\t", i*dt*2*ITPERFRAME);
		fprintf(fp1, "%f\t", (float)sol[i].e0);
    	fprintf(fp1, "%f\n", (float)sol[i].e1);
		}

	fclose (fp1);

	printf("Voltage in time file %d created\n", (int)floor(count*dt));

}


void printTip(std::vector<int> &dsizeTip, const char strA[], int x) {

  char strC1[128];
  char strC2[128];
  char strC3[128];
  const char *strB1 = "dataTip.dat";
  const char *strB2 = "dataGrad.dat";
  const char *strB3 = "dataTipSize.dat";

  strncpy(strC1,strA,x);
  strncpy(strC2,strA,x);
  strncpy(strC3,strA,x);
  strC1[x] = '\0';
  strC2[x] = '\0';
  strC3[x] = '\0';
  strcat(strC1,strB1);
  strcat(strC2,strB2);
  strcat(strC3,strB3);
  strcat(strC1,strA+x);
  strcat(strC2,strA+x);
  strcat(strC3,strA+x);

  //Print data
  FILE *fp1, *fp2, *fp3;
  fp1 = fopen(strC1,"w+");
  fp2 = fopen(strC2,"w+");
  fp3 = fopen(strC3,"w+");

  int dsize;
  cudaMemcpyFromSymbol(&dsize, dev_count, sizeof(int));

  if (dsize >= NN) {printf("OVERFLOW ERROR\n");}
  std::vector<vec3dyn> results1(dsize);
  std::vector<vec6dyn> results2(dsize);
  cudaMemcpyFromSymbol(&(results1[0]), dev_data1, dsize*sizeof(vec3dyn));
  cudaMemcpyFromSymbol(&(results2[0]), dev_data2, dsize*sizeof(vec6dyn));

  if (dsize > 1) {
    for (size_t i = 0;i<dsize;i++) {
      fprintf(fp1,"%f\t %f\t %f\n",results1[i].x,results1[i].y,results1[i].z);
      fprintf(fp2,"%f\t %f\t %f\t %f\t %f\t %f\n",
      	results2[i].x,results2[i].y,results2[i].z,results2[i].vx,results2[i].vy,results2[i].vz);
    }

    for (size_t i = 0;i<(dsizeTip.size());i++) {
      if (dsizeTip[i] > 0) {fprintf(fp3,"%d\n", dsizeTip[i]);}
    }
  }

  fclose (fp1);
  fclose (fp2);
  fclose (fp3);

}

void printParameters(paramVar param, const char strA[], int x) {

	char strC[128];
	const char *strB = "dataparam.dat";

	strncpy(strC,strA,x);
	strC[x] = '\0';
	strcat(strC,strB);
	strcat(strC,strA+x);

	//Print data
	FILE *fp1;
	fp1 = fopen(strC,"w+");

	fprintf(fp1,"Initial condition source: %s\n", param.initDataName);

	fprintf(fp1,"\n********Grid dimensions*********\n");
	fprintf(fp1,"# grid points X = %d\n", nx);
	fprintf(fp1,"# grid points Y = %d\n", ny);
	fprintf(fp1,"# grid points Z = %d\n", nz);
	fprintf(fp1,"Total number of nodes: %d\n", param.totpoints);

	fprintf(fp1,"\n********Spatial dimensions*********\n");
	fprintf(fp1,"Physical dx %f cm \n", (float)param.hx);
	fprintf(fp1,"Physical dy %f cm \n", (float)param.hy);
	fprintf(fp1,"Physical dz %f cm \n", (float)param.hz);
	fprintf(fp1,"Physical Lx length %f cm \n", (float)param.Lx);
	fprintf(fp1,"Physical Ly length %f cm \n", (float)param.Ly);
	fprintf(fp1,"Physical Lz length %f cm \n", (float)param.Lz);

	fprintf(fp1,"\n********Diffusion*********\n");
	fprintf(fp1,"Diffusion parallel component: %f cm^2/ms\n", (float)param.diff_par);
	fprintf(fp1,"Diffusion perpendicular component: %f cm^2/ms\n", (float)param.diff_per);
	fprintf(fp1,"Diffusion Dxx: %f cm^2/ms\n", (float)param.Dxx);
	fprintf(fp1,"Diffusion Dyy: %f cm^2/ms\n", (float)param.Dyy);
	fprintf(fp1,"Diffusion Dzz: %f cm^2/ms\n", (float)param.Dzz);

	#ifdef PERIODIC_Z
		fprintf(fp1,"\n******Periodic boundary conditions in Z*******\n");
	#endif

	#ifdef ANISOTROPIC_TISSUE
		fprintf(fp1,"\n******Anisotropic tissue*******\n");
		fprintf(fp1,"Diffusion Dxy: %f cm^2/ms\n", (float)param.Dxy);
		fprintf(fp1,"Initial fiber angle: %f deg\n", (float)param.initTheta);
		fprintf(fp1,"Total fiber rotation angle: %f deg\n", (float)param.d_theta);
		fprintf(fp1,"Fiber rotation rate: %f deg/mm\n", (float)param.d_theta/((float)param.Lz*10.f));
	#else
		fprintf(fp1,"\n******Isotropic tissue*******\n");
		fprintf(fp1,"rx (Dxx*dt/(dx*dx)): %f \n", (float)param.rx);
		fprintf(fp1,"ry (Dyy*dt/(dy*dy)): %f \n", (float)param.ry);
		fprintf(fp1,"rz (Dzz*dt/(dz*dz)): %f \n", (float)param.rz);
	#endif

	fprintf(fp1,"\n*****Time series******\n");
	fprintf(fp1,"Time step: %f ms\n", param.dt);
	fprintf(fp1,"Electrode position x: %f cm\n", (float)param.singlePoint_cm.x);
	fprintf(fp1,"Electrode position y: %f cm\n", (float)param.singlePoint_cm.y);
	fprintf(fp1,"Electrode position z: %f cm\n", (float)param.singlePoint_cm.z);

	fprintf(fp1,"\n********Time & performance*********\n");
	fprintf(fp1,"FPS:                    %f\n", (float)param.fpsCount);
	fprintf(fp1,"Total number of frames: %d\n", param.frameCount);
	fprintf(fp1,"Physical time:          %f ms\n", (float)param.physicalTime);
	fprintf(fp1,"Total time (real life): %f s \n", (float)param.tiempo);
	fprintf(fp1,"Iterations per frame: %d\n", ITPERFRAME);

	fprintf(fp1,"\n********Initial condition*********\n");
	fprintf(fp1,"Final iteration count: %d\n", param.count);
	fprintf(fp1,"Maximum number of iterations: %d\n", param.countlim);
  	if (param.counterclock) fprintf(fp1,"Counterclock spin\n");
	if (param.clock) fprintf(fp1,"Clock spin\n");

	fprintf(fp1,"\n********Model parameters*********\n");
	fprintf(fp1,"g_fi:    %f\n", (float)g_fi);
	fprintf(fp1,"tau_r:   %f\n", (float)tau_r);
	fprintf(fp1,"tau_si:  %f\n", (float)tau_si);
	fprintf(fp1,"tau_o:   %f\n", (float)tau_o);
	fprintf(fp1,"tau_vp:  %f\n", (float)tau_vp);
	fprintf(fp1,"tau_v1n: %f\n", (float)tau_v1n);
	fprintf(fp1,"tau_v2n: %f\n", (float)tau_v2n);
	fprintf(fp1,"tau_wp:  %f\n", (float)tau_wp);
	fprintf(fp1,"tau_wn:  %f\n", (float)tau_wn);
	fprintf(fp1,"u_c:     %f\n", (float)theta_c);
	fprintf(fp1,"u_v:     %f\n", (float)theta_v);
	fprintf(fp1,"u_csi:   %f\n", (float)u_csi);
	fprintf(fp1,"tau_d (C_m/g_fi):   %f\n", (float)tau_d);
	fprintf(fp1,"K:       %f\n", (float)K);
	fprintf(fp1,"C_m:      %f\n", (float)C_m);
	fprintf(fp1,"\nFilament voltage threshold: %f Volts\n", (float)Uth);

	fclose (fp1);

	puts("Parameter file created");

	/*------------------------------------------------------------------------
	* Create CSV file
	*------------------------------------------------------------------------
	*/

	const char *strBb = "dataparam.csv";

	strncpy(strC,strA,x);
	strC[x] = '\0';
	strcat(strC,strBb);
	strcat(strC,strA+x);

	FILE *fp2;
	fp2 = fopen(strC,"w+");

	char text[] = " ";

	#ifdef LOAD_DATA
		strcpy(text, "Initial condition source: ");
		fprintf(fp2,"%s,%s\n", text, param.initDataName);
	#else
		strcpy(text, "Initial condition source: ");
		fprintf(fp2,"%s,%s\n", text, param.initDataName);
	#endif

	// ********Grid dimensions*********
	strcpy(text, "# grid points X");
	fprintf(fp2,"%s,%d\n", text, nx);
	strcpy(text, "# grid points Y");
	fprintf(fp2,"%s,%d\n", text, ny);
	strcpy(text, "# grid points Z");
	fprintf(fp2,"%s,%d\n", text, nz);
	strcpy(text, "Total number of nodes");
	fprintf(fp2,"%s,%d\n", text, param.totpoints);

	//********Spatial dimensions*********
	strcpy(text, "Physical dx (cm)");
	fprintf(fp2,"%s,%f\n", text, (float)param.hx);
	strcpy(text, "Physical dy (cm)");
	fprintf(fp2,"%s,%f\n", text, (float)param.hy);
	strcpy(text, "Physical dz (cm)");
	fprintf(fp2,"%s,%f\n", text, (float)param.hz);
	strcpy(text, "Physical Lx length (cm)");
	fprintf(fp2,"%s,%f\n", text, (float)param.Lx);
	strcpy(text, "Physical Ly length (cm)");
	fprintf(fp2,"%s,%f\n", text, (float)param.Ly);
	strcpy(text, "Physical Lz length (cm)");
	fprintf(fp2,"%s,%f\n", text, (float)param.Lz);


	// ********Diffusion*********
	strcpy(text, "Diffusion parallel component (cm^2/ms)");
	fprintf(fp2,"%s,%f\n", text, (float)param.diff_par);
	strcpy(text, "Diffusion perpendicular component (cm^2/ms)");
	fprintf(fp2,"%s,%f\n", text, (float)param.diff_per);
	strcpy(text, "Diffusion Dxx (cm^2/ms)");
	fprintf(fp2,"%s,%f\n", text, (float)param.Dxx);
	strcpy(text, "Diffusion Dyy (cm^2/ms)");
	fprintf(fp2,"%s,%f\n", text, (float)param.Dyy);
	strcpy(text, "Diffusion Dzz (cm^2/ms)");
	fprintf(fp2,"%s,%f\n", text, (float)param.Dzz);

	#ifdef PERIODIC_Z
		strcpy(text, "Periodic boundary conditions in Z");
		fprintf(fp2,"%s,%d\n", text , 1);
	#else
		strcpy(text, "Periodic boundary conditions in Z");
		fprintf(fp2,"%s,%d\n", text , 0);
	#endif

	#ifdef ANISOTROPIC_TISSUE
		// ******Anisotropic tissue*******
		strcpy(text, "Diffusion Dxy (cm^2/ms)");
		fprintf(fp2,"%s,%f\n", text , (float)param.Dxy);
		strcpy(text, "Initial fiber angle (deg)");
		fprintf(fp2,"%s,%f\n", text , (float)param.initTheta);
		strcpy(text, "Total fiber rotation angle (deg)");
		fprintf(fp2,"%s,%f\n", text , (float)param.d_theta);
		strcpy(text, "Fiber rotation rate (deg/mm)");
		fprintf(fp2,"%s,%f\n", text , (float)(param.d_theta/param.Lz));
				// ******Isotropic tissue*******
		strcpy(text, "rx (Dxx*dt/(dx*dx))");
		fprintf(fp2,"%s,%d\n", text , 0);
		strcpy(text, "ry (Dyy*dt/(dy*dy))");
		fprintf(fp2,"%s,%d\n", text , 0);
		strcpy(text, "rz (Dzz*dt/(dz*dz))");
		fprintf(fp2,"%s,%d\n", text , 0);
	#else
		// ******Anisotropic tissue*******
		strcpy(text, "Diffusion Dxy (cm^2/ms)");
		fprintf(fp2,"%s,%d\n", text , 0);
		strcpy(text, "Initial fiber angle (deg)");
		fprintf(fp2,"%s,%d\n", text , 0);
		strcpy(text, "Total fiber rotation angle (deg)");
		fprintf(fp2,"%s,%d\n", text , 0);
		strcpy(text, "Fiber rotation rate (deg/mm)");
		fprintf(fp2,"%s,%d\n", text , text, 0);
		// ******Isotropic tissue*******
		strcpy(text, "rx (Dxx*dt/(dx*dx))");
		fprintf(fp2,"%s,%f\n", text , (float)param.rx);
		strcpy(text, "ry (Dyy*dt/(dy*dy))");
		fprintf(fp2,"%s,%f\n", text , (float)param.ry);
		strcpy(text, "rz (Dzz*dt/(dz*dz))");
		fprintf(fp2,"%s,%f\n", text , (float)param.rz);
	#endif

	// *****Time series******
	strcpy(text, "Electrode position x (cm)");
	fprintf(fp2,"%s,%f\n", text , (float)param.singlePoint_cm.x);
	strcpy(text, "Electrode position y (cm)");
	fprintf(fp2,"%s,%f\n", text , (float)param.singlePoint_cm.y);
	strcpy(text, "Electrode position z (cm)");
	fprintf(fp2,"%s,%f\n", text , (float)param.singlePoint_cm.z);

	// ********Time & performance*********
	strcpy(text, "FPS");
	fprintf(fp2,"%s,%f\n", text , param.dt);
	strcpy(text, "Time step");
	fprintf(fp2,"%s,%f\n", text , (float)param.fpsCount);
	strcpy(text, "Total number of frames");
	fprintf(fp2,"%s,%d\n", text , param.frameCount);
	strcpy(text, "Physical time (ms)");
	fprintf(fp2,"%s,%f\n", text , (float)param.physicalTime);
	strcpy(text, "Total time (real life) (s)");
	fprintf(fp2,"%s,%f\n", text , (float)param.tiempo);
	strcpy(text, "Iterations per frame");
	fprintf(fp2,"%s,%d\n", text , ITPERFRAME);

	// ********Initial condition*********
	strcpy(text, "Final iteration count");
	fprintf(fp2,"%s,%d\n", text, param.count);
	strcpy(text, "Maximum number of iterations");
	fprintf(fp2,"%s,%d\n", text, param.countlim);
	strcpy(text, "Counterclock spin");
	if (param.counterclock) {
		fprintf(fp2,"%s,%d\n", text, 1);
	} else {
		fprintf(fp2,"%s,%d\n", text, 0);
	}
	strcpy(text, "Clock spin");
	if (param.clock) {
		fprintf(fp2,"%s,%d\n", text, 1);
	} else {
		fprintf(fp2,"%s,%d\n", text, 0);
	}

	// ********Model parameters*********
	strcpy(text, "g_fi");
	fprintf(fp2,"%s,%f\n", text, (float)g_fi);
	strcpy(text, "tau_r");
	fprintf(fp2,"%s,%f\n", text, (float)tau_r);
	strcpy(text, "tau_si");
	fprintf(fp2,"%s,%f\n", text, (float)tau_si);
	strcpy(text, "tau_o");
	fprintf(fp2,"%s,%f\n", text, (float)tau_o);
	strcpy(text, "tau_vp");
	fprintf(fp2,"%s,%f\n", text, (float)tau_vp);
	strcpy(text, "tau_v1n");
	fprintf(fp2,"%s,%f\n", text, (float)tau_v1n);
	strcpy(text, "tau_v2n");
	fprintf(fp2,"%s,%f\n", text, (float)tau_v2n);
	strcpy(text, "tau_wp");
	fprintf(fp2,"%s,%f\n", text, (float)tau_wp);
	strcpy(text, "tau_wn");
	fprintf(fp2,"%s,%f\n", text, (float)tau_wn);
	strcpy(text, "theta_c");
	fprintf(fp2,"%s,%f\n", text, (float)theta_c);
	strcpy(text, "theta_v");
	fprintf(fp2,"%s,%f\n", text, (float)theta_v);
	strcpy(text, "u_csi");
	fprintf(fp2,"%s,%f\n", text, (float)u_csi);
	strcpy(text, "tau_d");
	fprintf(fp2,"%s,%f\n", text, (float)tau_d);
	strcpy(text, "K");
	fprintf(fp2,"%s,%f\n", text, (float)K);
	strcpy(text, "C_m");
	fprintf(fp2,"%s,%f\n", text, (float)C_m);
	strcpy(text, "Filament voltage threshold (volts)");
	fprintf(fp2,"%s,%f\n", text, (float)Uth);

	fclose (fp2);

}
