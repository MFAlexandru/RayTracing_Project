#ifndef GEOMETRY_H
#define GEOMETRY_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include "glm\vec3.hpp"
#include "glm\vector_relational.hpp"
#include "glm\geometric.hpp"

// GOOD TO GO ??


glm::dvec3 each_product(const glm::dvec3 &A, const glm::dvec3 &B) {
	return { A.x * B.x, A.y * B.y, A.z * B.z };
}

// MODULE == LENGTH

__host__ __device__ void top_char(glm::dvec3 *A) {
	if (A->x > 255.0) A->x = 255.0;
	if (A->y > 255.0) A->y = 255.0;
	if (A->z > 255.0) A->z = 255.0;
}

#endif