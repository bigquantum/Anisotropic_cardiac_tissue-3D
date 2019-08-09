
#include "typedef3V-FK.h"

#include "./common/helper_math.h"
#include "./common/CudaSafeCall.h"

/*------------------------------------------------------------------------
* Simple 3D volume renderer
*------------------------------------------------------------------------
*/

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;

// 3D texture
texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex1;
// 1D transfer function texture
texture<float4, 1, cudaReadModeElementType> transferTex;

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

struct Ray {
    float3 o;   // origin
    float3 d;   // direction
};

extern "C"
void initCuda(cudaExtent volumeSize, VolumeType *h_volume) {

    /*------------------------------------------------------------------------
    * Create 3D array
    *------------------------------------------------------------------------
    */

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    CudaSafeCall(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    /*------------------------------------------------------------------------
    * Copy data to 3D array
    *------------------------------------------------------------------------
    */

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume,
      volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    CudaSafeCall(cudaMemcpy3D(&copyParams));

    /*------------------------------------------------------------------------
    * Set texture parameters
    *------------------------------------------------------------------------
    */

    // access with normalized texture coordinates
    tex1.normalized = true;
    // linear interpolation
    tex1.filterMode = cudaFilterModeLinear;
     // clamp texture coordinates
    tex1.addressMode[0] = cudaAddressModeClamp;
    tex1.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    CudaSafeCall(cudaBindTextureToArray(tex1, d_volumeArray, channelDesc));

    /*------------------------------------------------------------------------
    * Create transfer function texture
    *------------------------------------------------------------------------
    */

    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    CudaSafeCall(cudaMallocArray(&d_transferFuncArray, &channelDesc2,
      sizeof(transferFunc)/sizeof(float4), 1));
    CudaSafeCall(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc,
      sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    // access with normalized texture coordinates
    transferTex.normalized = true;
    // wrap texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;

    // Bind the array to the texture
    CudaSafeCall(cudaBindTextureToArray(transferTex, d_transferFuncArray,
      channelDesc2));

    //CudaSafeCall(cudaFree(h_volume));
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    CudaSafeCall(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}


__device__ int intersectBox(Ray r, float3 boxmin, float3 boxmax,
  float *tnear, float *tfar) {

    /*------------------------------------------------------------------------
    * Compute intersection of ray with all six bbox planes
    *------------------------------------------------------------------------
    */

    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    /*------------------------------------------------------------------------
    * Re-order intersections to find smallest and largest on each axis
    *------------------------------------------------------------------------
    */

    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    /*------------------------------------------------------------------------
    * Find the largest tmin and the smallest tmax
    *------------------------------------------------------------------------
    */

    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

/*------------------------------------------------------------------------
* Transform vector by matrix (no translation)
*------------------------------------------------------------------------
*/

__device__ float3 mul(const float3x4 &M, const float3 &v) {

    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

/*------------------------------------------------------------------------
* Transform vector by matrix with translation
*------------------------------------------------------------------------
*/

__device__ float4 mul(const float3x4 &M, const float4 &v) {

    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba) {

    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}


__global__ void d_render(int flag, uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale) {

    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    /*------------------------------------------------------------------------
    * Calculate eye ray in world space
    *------------------------------------------------------------------------
    */

    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    /*------------------------------------------------------------------------
    * Find intersection with box
    *------------------------------------------------------------------------
    */

    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    /*------------------------------------------------------------------------
    * March along ray from front to back, accumulating color
    *------------------------------------------------------------------------
    */

    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    for (int i=0; i<maxSteps; i++) {

      // read from 3D texture
      // remap position to [0, 1] coordinates
      float sample = tex3D(tex1, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
      //sample *= 64.0f;    // scale for 10-bit data

      // lookup in transfer function texture
      float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
      col.w *= density;

      // "under" operator for back-to-front blending
      //sum = lerp(sum, col, col.w);

      // pre-multiply alpha
      col.x *= col.w;
      col.y *= col.w;
      col.z *= col.w;
      // "over" operator for front-to-back blending
      sum = sum + col*(1.0f - sum.w);

      // exit early if opaque
      if (sum.w > opacityThreshold)
          break;

      t += tstep;

      if (t > tfar) break;

      pos += step;
    }

    switch (flag) {

    	case 1:
    		sum *= brightness;
    	break;
    	case 2:
        // Spiral filament
    		if (sum.x > 0.f && sum.y > 0.f && sum.z > 0.f) sum = make_float4(1.0f);
    	break;
        case 3:
            //sum *= brightness;
            if (sum.x > 0.f && sum.y > 0.f && sum.z > 0.f) sum = make_float4(0.f,1.f,0.f,1.0f);
        break;
    	default:
    	break;
	   }

    sum *= brightness;

    /*------------------------------------------------------------------------
    * Transform vector by matrix (no translation)
    *------------------------------------------------------------------------
    */

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

extern "C"
void render_kernel(int flag, dim3 gridSize, dim3 blockSize,
  uint *d_output, uint imageW, uint imageH,
  float density, float brightness, float transferOffset, float transferScale) {

  d_render<<<gridSize, blockSize>>>(flag, d_output, imageW, imageH, density,
    brightness, transferOffset, transferScale);
  CudaCheckError();
}

extern "C"
void setTextureFilterMode(bool bLinearFilter) {

  tex1.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;

}

extern "C"
void freeCudaBuffers(void) {

  CudaSafeCall(cudaFreeArray(d_volumeArray));
  CudaSafeCall(cudaFreeArray(d_transferFuncArray));
}
