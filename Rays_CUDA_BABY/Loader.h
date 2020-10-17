#include <iostream>
#include <list>
#include <vector>

#include "Rendering.h"
#include "Images.h"

void load_to_mem(std::list<Sphere> *spheres_aux,
				std::list<Triangle> *triangles_aux,
				std::vector<glm::vec3> *v_buff_aux, Camera *cam_aux, glm::vec3 center) {
	// INITIALISE SPHERESS
	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(0.83, 0.67, 0.24),
		// SPECULAR COLOR
		glm::vec3(1, 0.93, 0.67),
		// REFLECTIVE COLOR
		glm::vec3(1, 1, 1),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		500.0,
		// RAY
		5,
		// POSITION X Y Z
		glm::vec3(30, 30, 30)));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SPECULAR COLOR
		glm::vec3(220.0 / 255, 40.0 / 255, 120.0 / 255),
		// REFLECTIVE COLOR
		glm::vec3(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		100,
		// RAY
		2,
		// POSITION X Y Z
		center + cam_aux->v * 9.0f + cam_aux->w * 10.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SPECULAR COLOR
		glm::vec3(220.0 / 255, 40.0 / 255, 120.0 / 255),
		// REFLECTIVE COLOR
		glm::vec3(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		100,
		// RAY
		2,
		// POSITION X Y Z
		center + cam_aux->u * 9.0f + cam_aux->w * 10.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SPECULAR COLOR
		glm::vec3(220.0 / 255, 40.0 / 255, 120.0 / 255),
		// REFLECTIVE COLOR
		glm::vec3(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		100,
		// RAY
		2,
		// POSITION X Y Z
		center - cam_aux->u * 9.0f + cam_aux->w * 10.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SPECULAR COLOR
		glm::vec3(220.0 / 255, 40.0 / 255, 120.0 / 255),
		// REFLECTIVE COLOR
		glm::vec3(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		100,
		// RAY
		2,
		// POSITION X Y Z
		center - cam_aux->v * 9.0f + cam_aux->w * 10.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SPECULAR COLOR
		glm::vec3(30.0 / 255, 170.0 / 255, 80.0 / 255),
		// REFLECTIVE COLOR
		glm::vec3(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		10,
		// RAY
		2,
		// POSITION X Y Z
		center + cam_aux->v * 9.0f / sqrtf(2) - cam_aux->u * 9.0f / sqrtf(2) + cam_aux->w * 50.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SPECULAR COLOR
		glm::vec3(30.0 / 255, 170.0 / 255, 80.0 / 255),
		// REFLECTIVE COLOR
		glm::vec3(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		10,
		// RAY
		2,
		// POSITION X Y Z
		center + cam_aux->v * 9.0f / sqrtf(2) + cam_aux->u * 9.0f / sqrtf(2) + cam_aux->w * 50.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SPECULAR COLOR
		glm::vec3(30.0 / 255, 170.0 / 255, 80.0 / 255),
		// REFLECTIVE COLOR
		glm::vec3(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		10,
		// RAY
		2,
		// POSITION X Y Z
		center - cam_aux->v * 9.0f / sqrtf(2) + cam_aux->u * 9.0f / sqrtf(2) + cam_aux->w * 50.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SPECULAR COLOR
		glm::vec3(30.0 / 255, 170.0 / 255, 80.0 / 255),
		// REFLECTIVE COLOR
		glm::vec3(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// LUMINOSITY
		glm::vec3(0, 0, 0),
		// SHININESS
		10,
		// RAY
		2,
		// POSITION X Y Z
		center - cam_aux->v * 9.0f / sqrtf(2) - cam_aux->u * 9.0f / sqrtf(2) + cam_aux->w * 50.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(0, 0, 0),
		// SPECULAR COLOR
		glm::vec3(0, 0, 0),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// LUMINOSITY
		glm::vec3(1, 1, 1),
		// SHININESS
		1,
		// RAY
		2,
		// POSITION X Y Z
		glm::vec3(0, 0, 30)));



	// INITIALISE SPHERE LIGHTS
	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(0, 0, 0),
		// SPECULAR COLOR
		glm::vec3(0, 0, 0),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// LUMINOSITY
		glm::vec3(0, 0, 1),
		// SHININESS
		1,
		// RAY
		2,
		// POSITION X Y Z
		cam_aux->v * -24.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(0, 0, 0),
		// SPECULAR COLOR
		glm::vec3(0, 0, 0),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// LUMINOSITY
		glm::vec3(1, 0, 0),
		// SHININESS
		1,
		// RAY
		2,
		// POSITION X Y Z
		cam_aux->u * -19.0f + cam_aux->v * 19.0f));

	spheres_aux->push_back(Sphere(
		// COLOR
		glm::vec3(0, 0, 0),
		// SPECULAR COLOR
		glm::vec3(0, 0, 0),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// LUMINOSITY
		glm::vec3(0, 1, 0),
		// SHININESS
		1,
		// RAY
		2,
		// POSITION X Y Z
		cam_aux->u * 19.0f + cam_aux->v * 19.0f));


	// INITIALISE TRIANGLES
	triangles_aux->push_back(Triangle(
		// COLOR
		glm::vec3(1, 1, 1),
		// SPECULAR COLOR
		glm::vec3(1, 1, 1),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// SHININESS
		0.0,
		// POSITION A B C
		cam_aux->w * sqrtf(2) * -40.0f + cam_aux->u * 20.0f + cam_aux->v * -10.0f,
		cam_aux->w * sqrtf(2) * -40.0f + cam_aux->u * -20.0f + cam_aux->v * -10.0f,
		cam_aux->w * sqrtf(2) * -40.0f + cam_aux->u * 0.0f + cam_aux->v * 20.0f,
		v_buff_aux));

	triangles_aux->push_back(Triangle(
		// COLOR
		glm::vec3(1, 1, 1),
		// SPECULAR COLOR
		glm::vec3(1, 1, 1),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// SHININESS
		0.0,
		// POSITION A B C
		cam_aux->position + cam_aux->w * 0.2f + cam_aux->u * 200.0f + cam_aux->v * -200.0f,
		cam_aux->position + cam_aux->w * 0.2f + cam_aux->u * -200.0f + cam_aux->v * -200.0f,
		cam_aux->position + cam_aux->w * 0.2f + cam_aux->u * 0.0f + cam_aux->v * 200.0f,
		v_buff_aux));

	triangles_aux->push_back(Triangle(
		// COLOR
		glm::vec3(1, 1, 1),
		// SPECULAR COLOR
		glm::vec3(1, 1, 1),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// SHININESS
		500.0,
		// POSITION A B C
		cam_aux->w * 0.0f + cam_aux->u * -20.0f + cam_aux->v * 20.0f,
		cam_aux->w * sqrtf(2) * -40.0f + cam_aux->u * -20.0f + cam_aux->v * -10.0f,
		cam_aux->w * sqrtf(2) * -40.0f + cam_aux->u * 0.0f + cam_aux->v * 20.0f,
		v_buff_aux));

	triangles_aux->push_back(Triangle(
		// COLOR
		glm::vec3(1, 1, 1),
		// SPECULAR COLOR
		glm::vec3(1, 1, 1),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// SHININESS
		500.0f,
		// POSITION A B C
		cam_aux->w * sqrtf(2) * 40.0f * -1.0f + cam_aux->u * 20.0f + cam_aux->v * -10.0f,
		cam_aux->w * 0.0f + cam_aux->u * 20.0f + cam_aux->v * 20.0f,
		cam_aux->w * sqrtf(2) * 40.0f * -1.0f + cam_aux->u * 0.0f + cam_aux->v * 20.0f,
		v_buff_aux));

	triangles_aux->push_back(Triangle(
		// COLOR
		glm::vec3(1, 1, 1),
		// SPECULAR COLOR
		glm::vec3(1, 1, 1),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// SHININESS
		500.0f,
		// POSITION A B C
		cam_aux->w * sqrtf(2) * 40.0f * -1.0f + cam_aux->u * 20.0f + cam_aux->v * -10.0f,
		cam_aux->w * sqrtf(2) * 40.0f * -1.0f + cam_aux->u * -20.0f + cam_aux->v * -10.0f,
		cam_aux->w * 0.0f + cam_aux->v * -25.0f,
		v_buff_aux));

	triangles_aux->push_back(Triangle(
		// COLOR
		glm::vec3(1, 1, 1),
		// SPECULAR COLOR
		glm::vec3(1, 1, 1),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// SHININESS
		500.0f,
		// POSITION A B C
		cam_aux->w * 0.0f + cam_aux->u * -20.0f + cam_aux->v * 20.0f,
		cam_aux->w * 0.0f + cam_aux->u * 20.0f + cam_aux->v * 20.0f,
		cam_aux->w * sqrtf(2) * 40.0f * -1.0f + cam_aux->u * 0.0f + cam_aux->v * 20.0f,
		v_buff_aux));

	triangles_aux->push_back(Triangle(
		// COLOR
		glm::vec3(1, 1, 1),
		// SPECULAR COLOR
		glm::vec3(1, 1, 1),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// SHININESS
		500.0f,
		// POSITION A B C
		cam_aux->w * sqrtf(2) * 40.0f * -1.0f + cam_aux->u * 20.0f + cam_aux->v * -10.0f,
		cam_aux->w * 0.0f + cam_aux->u * 20.0f + cam_aux->v * 20.0f,
		cam_aux->w * 0.0f + cam_aux->v * -25.0f,
		v_buff_aux));

	triangles_aux->push_back(Triangle(
		// COLOR
		glm::vec3(1, 1, 1),
		// SPECULAR COLOR
		glm::vec3(1, 1, 1),
		// REFLECTIVE COLOR
		glm::vec3(0, 0, 0),
		// SHININESS
		500.0f,
		// POSITION A B C
		cam_aux->w * sqrtf(2) * 40.0f * -1.0f - cam_aux->u * 20.0f + cam_aux->v * -10.0f,
		cam_aux->w * 0.0f - cam_aux->u * 20.0f + cam_aux->v * 20.0f,
		cam_aux->w * 0.0f + cam_aux->v * -25.0f,
		v_buff_aux));
}
