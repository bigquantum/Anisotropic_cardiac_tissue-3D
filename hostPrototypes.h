
// Main program functions
void initGates(stateVar g_h, stateVar gOut_d, stateVar gIn_d, int memSize,
REAL *J_current_d);
void loadData(stateVar g_h);
void SIM_2V_wrapper(dim3 grid3D, dim3 block3D, stateVar gOut_d, stateVar gIn_d,
   conductionVar r_d, REAL *J_current_d);
void animation(dim3 grid3D, dim3 block3D,
  stateVar g_h, stateVar gOut_d, stateVar gIn_d, REAL *J_current_d,
  conductionVar r_d, paramVar param, REAL *pt_h, REAL *pt_d,
  std::vector<electrodeVar> &electrode,
  bool initConditionFlag);
void singlePoint(stateVar gIn_d, REAL *pt_h, REAL *pt_d,
   int singlePointPixel, std::vector<electrodeVar> &electrode);
void copyRender(dim3 grid1D, dim3 block1D, int totpoints,
  stateVar gIn_d, VolumeType *h_volume);
VolumeType *spiralTip(dim3 grid3Dz, dim3 block3Dz, REAL *v_past_d,
  stateVar gIn_d, VolumeType *h_volume);
void cutVoltage(paramVar p, stateVar g_h, stateVar g_present_d);
void stimulateV(int memSize, stateVar g_h, stateVar g_present_d);
int initSinglePoint(float3 sp, paramVar param);
void exitProgram(void);

// Helper functions
void swapSoA(stateVar *A, stateVar *B);
void computeFPS(void);
void screenShot(int w, int h);
void chirality(int len_text, char text[], bool *counterclock, bool *clock);

// Save data functions
void print1D(stateVar g_h, int count, const char strA[], int x);
void print2D(stateVar g_h, int count, const char strA[], int x);
void print3D(stateVar g_h, int count, const char strA[], int x);
void printVoltageInTime(std::vector<electrodeVar> &sol, const char strA[],
  int x, REAL dt, int count);
void printTip(std::vector<int> &dsizeTip, const char strA[], int x);
void printParameters(paramVar param, const char strA[], int x);

// Graphics functions
extern "C"
void initCuda(cudaExtent volumeSize, VolumeType *h_volume);
extern "C"
void render_kernel(int flag, dim3 gridSize, dim3 blockSize,
  uint *d_output, uint imageW, uint imageH,
  float density, float brightness, float transferOffset, float transferScale);
extern "C"
void setTextureFilterMode(bool bLinearFilter);
extern "C"
void freeCudaBuffers();
extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
