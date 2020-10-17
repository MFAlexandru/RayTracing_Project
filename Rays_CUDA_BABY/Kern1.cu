#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <sm_60_atomic_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cuda.h>

#include <iostream>
#include <list>
#include <vector>

#include "Rendering.h"
#include "Images.h"
#include "Loader.h"

#include "SFML-2.5.1\include\SFML\Graphics.hpp"
#include "SFML-2.5.1\include\SFML\OpenGL.hpp"
#include <stdio.h>

#define cudaDeviceScheduleBlockingSync 0x04
// EXPERIMENTAL

__device__ glm::vec3 colision(Sphere* spheres, Triangle* triangles, glm::vec3* Vbuff, glm::ivec2 &sizes, float MAX_T, glm::vec3 &p, glm::vec3 &d, int light_pos) {

	float min = INFINITY;
	float t;
	int index = -1;
	index ;
	int k = 0;
	
	for (int i = 0; i <sizes.x; i++) {
		
		t = spheres[i].Hit(p, d);
		if (t > 0 && t < MAX_T && t < min && i != light_pos) {
			min = t;
			index = i;
		}
	}
	for (int i = 0; i < sizes.y; i++) {
		t = triangles[i].Hit(p, d, Vbuff);
		if (t > 0 && t < MAX_T && t < min) {
			min = t;
			index = i;
			k = 1;
		}
	}

	return {min, index, k};
}

// glm::vec3 ADDITIONS

__device__ bool operator<(glm::vec3 &A, glm::vec3 &B) {
	if (A.x < B.x && A.y < B.y && A.z < B.z) return true;
	else return false;
}

// END OF ADDITIONS

__device__ glm::vec3 ray_color(Sphere* spheres, Triangle* triangles, glm::vec3* Vbuff, glm::ivec3 &sizes, glm::vec3 p, glm::vec3 d, curandState* state, int &bounce) {
	glm::vec3 Pixel(0, 0, 0), scale(1, 1, 1), zero(0, 0, 0), x(1, 0, 0), y(0, 1, 0), z(0, 0, 1), epsilon(0.0001f, 0.00001f, 0.00001f);
	glm::vec3 specular, color, reflect, luminosity;
	glm::vec3 n, p1, d1, h, ray, colide, light_pos;
	float T_MAX, t, t1, shininess;
	int index, light_count = 5;

	for (int q = 0; q <= bounce && epsilon < scale; q++) {
		colide = colision(spheres, triangles, Vbuff, glm::ivec2(sizes.x, sizes.y), INFINITY, p, d, -1);
		t = colide.x;
		index = (int)colide.y;
		if (t >= 0 && t < INFINITY) {
			// Copy data
			{
			if ((int)colide.z == 0) {
				color = spheres[index].color;
				reflect = spheres[index].reflect;
				specular = spheres[index].specular;
				shininess = spheres[index].shininess;
				luminosity = spheres[index].luminosity;
				n = spheres[index].get_normal(d, p, t);
			}
			else {
				color = triangles[index].color;
				reflect = triangles[index].reflect;
				specular = triangles[index].specular;
				shininess = triangles[index].shininess;
				luminosity = { 0, 0, 0 };
				n = triangles[index].get_normal(d, p, t, Vbuff);
			}
			}


			Pixel = Pixel + scale * (color / 8.0f + luminosity);

			// PROBLEMA: DACA NU MA OPRESC AICI BILELE DE LUMINA APAR NEGRE DESI NU AR TREBUI ???

			for (int i = 0; i < sizes.x; i++) {
				if (spheres[i].luminosity == zero) continue;
				for (int j = 0; j < light_count; j++) {
					// CALCULAREA P1 D1 CE REPREZINTA PARAMETRII RAZEI DE
					// LA OBIECT LA LUMMINA
					// MAI INTAI CALCULAM NOILE PUNCTE DE LUMINA
					light_pos = spheres[i].position + (curand_uniform(state) * x + curand_uniform(state) * y + curand_uniform(state) * z) * spheres[i].radius;
					p1 = point_at(p, d, t);
					// d1 = light_pos + curand_uniform(state) - p1; asa era inainte da nu stiu dc
					d1 = light_pos - p1;
					d1 = glm::normalize(d1);
					p1 = p1 + (d1 * 0.1f);
					T_MAX = glm::length(light_pos - p1);

					colide = colision(spheres, triangles, Vbuff, glm::ivec2(sizes.x, sizes.y), T_MAX, p1, d1, i);
					t1 = colide.x;

					// VEDEM DACA CEVA BLOCHEAZA LUMINA
					if (t1 < 0 || t1 > T_MAX) {
						h = normalize(d1 - d);
						Pixel = Pixel +
							scale * (specular * (spheres[i].luminosity / (float)light_count) * powf(fmaxf(0.0, glm::dot(n, h)), shininess) +
								color * (spheres[i].luminosity / (float)light_count) * fmaxf(0.0, glm::dot(n, d1)));
					}
				}
			}
			ray = glm::normalize(glm::reflect(d, n));
			p = point_at(p + (ray * 0.1f), d, t);
			d = ray;
			scale = reflect * scale;
		}

	}

	return Pixel;
}

__device__ void Put_Pixel(int &width, unsigned char *data, glm::vec3 p, int x, int y) {
	data[(x * width + y) * 3] = (unsigned char)p.x;
	data[(x * width + y) * 3 + 1] = (unsigned char)p.y;
	data[(x * width + y) * 3 + 2] = (unsigned char)p.z;
}

// In urmatoarea fct i si j sunt momentan inutili

__global__ void atempt(Sphere* spheres, Triangle* triangles, glm::vec3* Vbuff, glm::ivec3 sizes, unsigned char *data, int height, int width, Camera *cam, int i, int j, int AA, int bounce, curandState *state, glm::ivec3 mem_size) {
	

	// SHARED MEMORY COPY
	extern __shared__ char total[];
	Sphere* shr_sphere = (Sphere*)(total);
	Triangle* shr_triangle = (Triangle*)(total + mem_size[0]);
	glm::vec3* shr_Vbuff = (glm::vec3*)(total + mem_size[0] + mem_size[1]);
	if (threadIdx.x < sizes[0])  shr_sphere[threadIdx.x] = spheres[threadIdx.x];
	if (threadIdx.x < sizes[1])  shr_triangle[threadIdx.x] = triangles[threadIdx.x];
	if (threadIdx.x < sizes[2])  shr_Vbuff[threadIdx.x] = Vbuff[threadIdx.x];
	__syncthreads();


	// BEGIN RENDER
	glm::vec3 Pixel(0, 0, 0), p2, dir, eye;
	curandState local_state = state[threadIdx.x * blockDim.x + threadIdx.y];
	if (!(blockIdx.x * blockDim.x + threadIdx.x > height || blockIdx.y * blockDim.x + threadIdx.y > width)) {
		for (int k = 0; k < AA; k ++)
			for (int q = 0; q < AA; q++) {
				eye = cam->eye_to(&local_state);
				dir = cam->direct_to(blockIdx.x * blockDim.x + threadIdx.x + (float)k / (float)AA, blockIdx.y * blockDim.x + threadIdx.y + (float)q / (float)AA, eye);
				Pixel = Pixel + ray_color(shr_sphere, shr_triangle, shr_Vbuff, sizes, eye, dir, &local_state, bounce) * 256.0f;
			}
		Pixel = Pixel / (float)(AA * AA);
		top_char(&Pixel);
		Put_Pixel(width, data, Pixel, blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.x + threadIdx.y);
	}
	state[threadIdx.x * blockDim.x + threadIdx.y] = local_state;
	__syncthreads();
}


__global__ void kernel_setup(curandState *state) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}

int main() {
	// NU MERGE CU NIMIC MAI MIC DE 900 PE 900 NU STIU DE CE NU SE SINCRONIZEAZA
	// SET PHOTO PARAMETERS
	int h = 4000, w = 4000, bounce = 10, threadSize = 22, antialiasing = 10;
	glm::vec3 center(30, 30, 30);


	// SET THE DEVICE
	int numDevice = 0;
	int maxDevice = 0;
	cudaGetDeviceCount(&numDevice);
	if (numDevice > 1) {
		int MaxMultiprocessors = 0;
		for (int device = 0; device < numDevice; device++) {
			cudaDeviceProp props;
			cudaGetDeviceProperties(&props, device);
			if (MaxMultiprocessors < props.multiProcessorCount) {
				MaxMultiprocessors = props.multiProcessorCount;
				maxDevice = device;
			}
		}
	}
	cudaSetDevice(maxDevice);
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

	// CURAND VARIABLES SETUP
	curandState *state;
	cudaMalloc(&state, sizeof(curandState) * threadSize * threadSize);
	cudaMemset(state, 0, threadSize * threadSize * sizeof(curandState));
	kernel_setup << <threadSize, threadSize>> > (state);


	// DEVICE IMAGE INITIALISATION
	img::Image dev_img(h, w);
	unsigned char* data;
	cudaMalloc(&data, sizeof(unsigned char) * h * w * 3);


	// CPU MEMORY ALOCATION
	std::list<Sphere> spheres_aux;
	std::list<Triangle> triangles_aux;
	std::vector<glm::vec3> v_buff_aux;


	// INITIALISE AND COPY CAMERA TO DEVICE
	Camera cam_aux(
		// CAMERA POSITION
		glm::vec3(-10, -10, -10),
		center - glm::vec3(-10, -10, -10),
		// DISTANCE FROM CAMERA FOCUS
		40 * sqrt(2) * 1.2,
		// CAMERA ANGLE
		3.14 / 2.5,
		// Focus amount
		0.01,
		// PICTURE SIZE
		h,
		w);
	Camera *cam;
	cudaMalloc(&cam, sizeof(Camera));
	cudaMemcpy(cam, &cam_aux, sizeof(Camera), cudaMemcpyHostToDevice);


	// LOAD SCENE TO CPU
	load_to_mem(&spheres_aux, &triangles_aux, &v_buff_aux, &cam_aux, center);


	// COPPY TO DEVICE
	thrust::device_vector<Sphere> spheres(spheres_aux.begin(), spheres_aux.end());
	thrust::device_vector<Triangle> triangles(triangles_aux.begin(), triangles_aux.end());
	thrust::device_vector<glm::vec3> v_buff(v_buff_aux.begin(), v_buff_aux.end());
	std::cout << 1;


	// SET DEVICE HEAP SIZE
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4 * 4001 * 4001);


	// START RENDER
	
	int scale = ceil((float)h / (float)threadSize);
	atempt << <dim3(scale, scale), dim3(threadSize, threadSize), sizeof(Sphere) * spheres.size() + sizeof(Triangle) * triangles.size() + sizeof(glm::vec3) * v_buff.size()>> > (
		thrust::raw_pointer_cast(&spheres[0]),
		thrust::raw_pointer_cast(&triangles[0]),
		thrust::raw_pointer_cast(&v_buff[0]),
		glm::ivec3(spheres.size(), triangles.size(), v_buff.size()),
		data, h, w, cam, 0, 0, antialiasing, bounce, state,
		glm::ivec3(sizeof(Sphere) * spheres.size(), sizeof(Triangle) * triangles.size(), sizeof(glm::vec3) * v_buff.size()));
	cudaDeviceSynchronize();


	// COPY RESULT TO CPU
	cudaMemcpy(dev_img.data, data, sizeof(unsigned char) * 3 * h * w, cudaMemcpyDeviceToHost);
	dev_img.Image_write("file.png");
	std::cout << 2;
	cudaFree(data);
}

