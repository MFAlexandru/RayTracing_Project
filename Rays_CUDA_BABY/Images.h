#ifndef IMAGES_LIB_H
#define IMAGES_LIB_H

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdlib.h> 
#include <string.h>

#include "glm\vec3.hpp"
#include "glm\vector_relational.hpp"
#include "glm\mat3x3.hpp"
#include "glm\matrix.hpp"

#include "stb-master/stb_image.h"
#include "stb-master/stb_image_write.h"
namespace img {
	struct Image {
	public:

		int height, width;
		unsigned char* data;

		Image() :height(0), width(0), data(0) {};
		Image(int H, int W);
		~Image();
		void Put_Pixel(glm::dvec3 P, int x, int y);
		glm::dvec3 get_Pixel(int x, int y);
		Image(const char* file, const int & H, const int & W);
		int get_height() {
			return height;
		};
		int get_width() {
			return width;
		};
		void Image_write(char const* file) {
			stbi_write_png(file, width, height, 3, data, sizeof(unsigned char) * 3 * width);

		};
	};

	Image::Image(int H, int W) {
		height = H;
		width = W;
		data = new unsigned char[H * W * 3];
	}

	// The pixel range is 0 <= pos < get_height / get_width
	__host__ __device__ void Image::Put_Pixel(glm::dvec3 p, int x, int y) {
		data[(x * width + y) * 3] = (unsigned char)p.x;
		data[(x * width + y) * 3 + 1] = (unsigned char)p.y;
		data[(x * width + y) * 3 + 2] = (unsigned char)p.z;
	}

	glm::dvec3 Image::get_Pixel( int x, int y) {
		return { (double) data[(x * width + y) * 3],
			(double) data[(x * width + y) * 3 + 1],
			(double) data[(x * width + y) * 3 + 2] };
	}
	// Outputs the image to a PNG File


	Image::~Image() {
		delete data;
	}

}


// DEFINITIONS # DEFINITIONS # DEFINITIONS # DEFINITIONS # DEFINITIONS


#endif