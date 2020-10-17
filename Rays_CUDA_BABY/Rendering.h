#pragma once
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand_kernel.h>

#include "glm\vec3.hpp"
#include "glm\vector_relational.hpp"
#include "glm\mat3x3.hpp"
#include "glm\matrix.hpp"

struct Light {
	glm::vec3 pos;
	glm::vec3 color;
};


struct Camera {
	// 'position' == 'e'
	glm::vec3 position;
	glm::vec3 w, u, v;
	float d, l;
	const int nx, ny;
	float lens, lens_distance;

	__host__ __device__ Camera(
		glm::vec3 e,
		glm::vec3 Vec,
		float d,
		float angle,
		float LENS_DISTANCE,
		int nx,
		int ny)
		: v(0, 0, 1), d(d), nx(nx), ny(ny), lens_distance(LENS_DISTANCE)
	{
		l = 2 * d / tan(angle);
		lens = lens_distance * l;
		w = -Vec;
		position = e;
		w = normalize(w);
		if (abs(dot(w, v)) != 1) {
			u = cross(v,w);
			v = cross(w, u);
			v = normalize(v);
			u = normalize(u);
		}
		else {
			u = { 1, 0, 0 };
			v = cross(w, -u);
			u = cross(w, v);
			v = normalize(v);
			u = normalize(u);
		}
	};
	__device__ glm::vec3 eye_to(curandState *state)
	{
		glm::vec3 eye;
		eye = position - u * (lens / 2.0f - lens * curand_uniform(state))
			+ v * (lens / 2.0f - lens * curand_uniform(state))
			- w * lens_distance * d;
		return eye;
	}

	__device__ glm::vec3 direct_to(float i, float j, glm::vec3 eye)
	{
		glm::vec3 direct;
		direct= position - eye + w * (-1.0f) * d +
			v * (l / 2.0f - l * (i + 0.5f) / nx) +
			u * (0.0f - l / 2.0f + l * (j + 0.5f) / ny);
		direct = normalize(direct);
		return direct;
	}

};

using vector3f = glm::vec3;
using vector3d = glm::vec3;
using vector3d_ref = const glm::vec3&;

// COMPUTE FUNCTIONS
/*
__device__ inline glm::vec3 vec_add(glm::vec3 &a, glm::vec3 &b) {
	return {
		__dadd_rn(a.x, b.x),
		__dadd_rn(a.y, b.y),
		__dadd_rn(a.z, b.z)
	};
}

__device__ inline glm::vec3 vec_mul(glm::vec3 a, glm::vec3 b) {
	return {
		__dmul_rn(a.x, b.x),
		__dmul_rn(a.y, b.y),
		__dmul_rn(a.z, b.z)
	};
}
*/

__device__ inline glm::vec3 point_at(vector3d_ref p, vector3d_ref d, float t) {
	return p + d * t;
}


struct Sphere {

	glm::vec3 color;
	glm::vec3 specular;
	glm::vec3 reflect;
	glm::vec3 luminosity;
	float radius;
	float shininess;
	glm::vec3 position;
	
	__host__ __device__ Sphere() = default;
	__host__ __device__ ~Sphere() = default;
	__host__ __device__ Sphere(glm::vec3 C, glm::vec3 S, glm::vec3 REF, glm::vec3 LUM, float P, float R, glm::vec3 POS)
	{
		radius = R;
		shininess = P;
		luminosity = LUM;
		position = POS;
		color = C;
		specular = S;
		reflect = REF;
	}

	__device__ glm::vec3 get_normal(vector3d_ref d, vector3d_ref p, float t)
	{
		glm::vec3 n = point_at(p, d, t) - position;
		n = normalize(n);
		return n;
	}
	__device__ float Hit(vector3d_ref p, vector3d_ref d)
	{
		//float Delta, T1, T2;
		
		auto ecdiff = p - position;
		auto d2 = dot(d, d);
		auto d_ecdiff = dot(d, ecdiff);

		auto Delta = d_ecdiff * d_ecdiff - d2 * (dot(ecdiff, ecdiff) - radius * radius);
		if (Delta < 0) {
			return NAN;
		}
		else if (Delta == 0) {
			return -d_ecdiff / d2;
		}
		else {
			auto d_sqrt = sqrt(Delta);
			auto T1 = (-d_ecdiff + d_sqrt) / d2;
			auto T2 = (-d_ecdiff - d_sqrt) / d2;
			return fmin(T1, T2);
		}
	}
	
};

__device__ float DET(vector3d_ref a, vector3d_ref b, vector3d_ref d1) {
	//glm::mat3 M(a, b, d1);
	
	float M = b.y * (a.x * d1.z - d1.x * a.z) +
		b.z * (a.y * d1.x - d1.y * a.x) +
		b.x * (a.z * d1.y - d1.z * a.y);

	//return glm::determinant(M);
	return M;
}
__device__ void top_char(glm::vec3 *A) {
	if (A->x > 255.0) A->x = 255.0;
	if (A->y > 255.0) A->y = 255.0;
	if (A->z > 255.0) A->z = 255.0;
}

struct Triangle {
	glm::vec3 color;
	glm::vec3 specular;
	glm::vec3 reflect;
	//glm::vec3 normal;
	int a1, b1, c1;
	float shininess;
	__host__ __device__ Triangle() = default;
	__host__ __device__ ~Triangle() = default;
	__host__ Triangle(vector3d COLOR,
		vector3d S,
		vector3d REF,
		float P,
		vector3d A,
		vector3d B,
		vector3d C,
		std::vector<glm::vec3> *Buffer
	) {
		//normal = cross((B - A), C - A);
		//normal = normalize(normal);
		std::vector<glm::vec3>::iterator iter;
		color = COLOR;
		specular = S;
		reflect = REF;
		shininess = P;
		iter = find(Buffer->begin(), Buffer->end(), A);
		if (iter != Buffer->end()) {
			a1 = iter - Buffer->begin();
		}
		else {
			Buffer->push_back(A);
			a1 = Buffer->size() - 1;
		}

		iter = find(Buffer->begin(), Buffer->end(), B);
		if (iter != Buffer->end()) {
			b1 = iter - Buffer->begin();
		}
		else {
			Buffer->push_back(B);
			b1 = Buffer->size() - 1;
		}

		iter = find(Buffer->begin(), Buffer->end(), C);
		if (iter != Buffer->end()) {
			c1 = iter - Buffer->begin();
		}
		else {
			Buffer->push_back(C);
			c1 = Buffer->size() - 1;
		}
	}
	
	__device__ glm::vec3 get_normal(vector3d_ref d, vector3d_ref p, float t, glm::vec3* Buffer) 
	{
		glm::vec3 n = cross((Buffer[b1] - Buffer[a1]), Buffer[c1] - Buffer[a1]);
		n = normalize(n);
		if (dot(n, -d) < 0)
			n = -n;
		return n;
	}
	__device__ float Hit(vector3d_ref p, vector3d_ref d1, glm::vec3* Buffer) 
	{
		glm::vec3 x = Buffer[a1] - Buffer[b1], y = Buffer[a1] - Buffer[c1], a_e = Buffer[a1] - p;
		float M = DET(x, y, d1), t, beta, gama;
		t = DET(x, y, a_e) / M;
		gama = DET(x, a_e, d1) / M;
		if (gama < 0 || gama > 1)
			return NAN;
		beta = DET(a_e, y, d1) / M;
		if (beta < 0 || beta > 1 - gama)
			return NAN;
		return t;
	}
	
};
