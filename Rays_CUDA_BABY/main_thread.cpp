
#include <iostream>
#include <thread>
#include <atomic>
#include <queue>
#include <memory>

#include <time.h>

//#include "Images.h"
#include "Geometry.h"
#include "Rendering.h"
#include "Handle.h"

#include "SFML\Graphics.hpp"

#include "thrust/host_vector.h"

struct Scene {
	std::vector<std::unique_ptr<Surface>> objs;
	std::vector<Light> lights;
};

// auto [t, index] = call(); // t de la timp?
std::pair<double, int> colision(const Scene& scene, double MAX_T, vector3<double> p, vector3<double> d) {
	double min = INFINITY;
	double t;
	int index = -1;
	//int i = 0;
	//for (const auto& ob : scene.objs) {
	for (int i = 0; i < scene.objs.size(); i++){
		t = scene.objs[i]->Hit(p, d);
		if (t > 0 && t < MAX_T && t < min) {
			min = t;
			index = i;
		}
		//i++;
	}

	std::pair<double, int> out;
	if (INFINITY == min)
		out.first = NAN;
	else
		out.first = min;
	out.second = index;
	return out;
}

// p-punct, d-directia
vector3<double> ray_color(const Scene& scene, const vector3<double>& p, const vector3<double>& d, int bounce) {
	vector3<double> Pixel(0, 0, 0);
	if (bounce == -1) 
		return Pixel;

	vector3<double> n, p1, d1, h, ray, specular(1, 1, 1);

	double T_MAX;

	auto [t, index] = colision(scene , INFINITY, p, d);

	if (t >= 0) {
		Pixel = scene.objs[index]->color / 8;
		// AM LOVIT UN OBIECT DECI TEBUIE SA VERIFICAM DACA VEDEM
		// O SURSA DE LUMINA
		n = scene.objs[index]->get_normal(d, p, t);

		for(const Light& light : scene.lights) {
			// CALCULAREA P1 D1 CE REPREZINTA PARAMETRII RAZEI DE
			// LA OBIECT LA LUMMINA
			p1 = point_at(p, d, t);
			d1 = light.pos - p1;
			d1.normalise();
			p1 = p1 + d1 * 0.1;
			T_MAX = (light.pos - p1).module();

			auto [t1, _] = colision(scene, T_MAX, p1, d1);

			// VEDEM DACA A
			if (!(t1 > 0)) { //if( t1 <= 0) {  /*asta nu merge???*/
				h = d1 - d;
				h.normalise();

				Pixel +=
					scene.objs[index]->specular.each_product(light.color) * pow(std::max(0.0, n * h), scene.objs[index]->shininess) +
					scene.objs[index]->color.each_product(light.color) * std::max(0.0, n * d1);
			}
		}
		// ORIGINAL
		ray = d - n * 2 * (d * n);
		ray.normalise();
		Pixel += scene.objs[index]->reflect.each_product(ray_color(scene, point_at(p + ray * 0.1, d, t), ray, bounce - 1));
		// MY WIERD VERSION
		//Pixel = Pixel.max(obj[index]->get_ref().each_product(ray_color(point_at(p + r * 0.1, d, t), r, bounce - 1)));
	}
	return Pixel;
}

vector3<double> get_eye(Camera cam) {
	return cam.position - cam.u * (cam.lens / 2 - cam.lens * (double)rand() / RAND_MAX)
		+ cam.v * (cam.lens / 2 - cam.lens * (double)rand() / RAND_MAX)
		- cam.w * cam.lens_distance;
}

void line_render(const Scene& scene, sf::Image* Img, int bounce, Camera cam, std::atomic<int>* progress, int lowerBound, int upperBound, int samples) {
	vector3<double> p, d, Pixel(0, 0, 0), increment_l, increment_h, base_d;
	sf::Color Prev_pixel;
	increment_l = cam.u * (cam.l * 1.0 / cam.nx / samples / 10);
	increment_h = cam.v * (cam.l * 1.0 / cam.ny / samples * -1 / 10);
	for (int l = 0; l < samples; l++) {
		for (int h = 0; h < samples; h++) {
			for(int i=lowerBound; i<upperBound; i++){
				progress->fetch_add(1);
				for (int j = 0; j < Img->getSize().x; j++) {
					Prev_pixel = Img->getPixel(j, i);
					Pixel = { (double)Prev_pixel.r, (double)Prev_pixel.g , (double)Prev_pixel.b };
					// Antialiasing iplementation
			
					cam.direct_to(i + 1.0 / samples * (l + (double)rand() / RAND_MAX),
						j + 1.0 / samples * (h + (double)rand() / RAND_MAX), rand);
					d = cam.direction;
					p = cam.eye;
					Pixel = (Pixel * (l * samples + h) + ray_color(scene, p,
					d
					, bounce) * 255) / (l * samples + h + 1);
					
					Pixel.top_char();
					Img->setPixel(j, i, sf::Color(Pixel.x, Pixel.y, Pixel.z, 255));
		}
	}
		}
	}
}

void window(const Scene& scene, sf::Image* Img, Camera cam, std::atomic<int>* progress) {
	sf::RenderWindow Window;
	Window.create(sf::VideoMode((unsigned int)cam.nx, (unsigned int)cam.ny), "Progress display");

	Window.setKeyRepeatEnabled(false);
	sf::Texture Tex;
	sf::Sprite Image_progress;
	

	while (Window.isOpen()) {
		sf::Event Event;
		while (Window.pollEvent(Event)) {
			switch (Event.type) {
			case sf::Event::Closed:
					Window.close();
					break;
			}

		}
		Tex.loadFromImage(*Img);
		Image_progress.setTexture(Tex);
		Window.draw(Image_progress);
		Window.display();
	}
}

/* credits: 
 * Tudor: rescris chestii, si optimizat
*/

int ain() {
	//int h, w, bounce;
	//std::cout << "Height:" << std::endl;
	//std::cin >> h;
	//std::cout << "Width:" << std::endl;
	//std::cin >> w;
	//std::cout << "Bounce:" << std::endl;
	//std::cin >> bounce;

	//int h = 4000, w = 4000, bounce = 30;
	//int h = 300, w = 300, bounce = 3;
	int h = 1000, w = 1000, bounce = 4;
	
	srand((unsigned)time(NULL));
	Scene scene;

	std::unique_ptr<Surface> tempPtr = std::make_unique<Sphere>();
	Sphere *SPH = dynamic_cast<Sphere*>(tempPtr.get());
	scene.objs.emplace_back(std::move(tempPtr));
	SPH->color = { 0.83, 0.67, 0.24 };
	SPH->specular = { 1, 0.93, 0.67 };
	SPH->reflect = { 1, 1, 1 };
	SPH->shininess = 500;
	SPH->radius = 5;
	SPH->position = { 30, 30, 30 };

	/*
	Sphere* SPH = new Sphere(
		// COLOR
		vector3<double>(0.83, 0.67, 0.24),
		// SPECULAR COLOR
		vector3<double>(1, 0.93, 0.67),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// RAY
		5,
		// POSITION X Y Z
		vector3<double>(30, 30, 30));
	*/

	Camera cam(
		// CAMERA POSITION
		vector3<double>(-10, -10, -10),
		SPH->get_c() - vector3<double>(-10, -10, -10),
		// DISTANCE FROM CAMERA FOCUS
		40 * sqrt(2) * 1.2,
		// CAMERA ANGLE
		3.14 / 2.5,
		// Focus amount
		0.02,
		// PICTURE SIZE
		h,
		w);



	// SPHERES # SPHERES # SPHERES # SPHERES # SPHERES # SPHERES #
	Surface* SPH2 = new Sphere(
		// COLOR
		vector3<double>(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SPECULAR COLOR
		vector3<double>(220.0 / 255, 40.0 / 255, 120.0 / 255),
		// REFLECTIVE COLOR
		vector3<double>(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SHININESS
		100,
		// RAY
		2,
		// POSITION X Y Z
		SPH->get_c() + cam.get_v() * 9 + cam.get_w() * 10);

	Sphere* SPH3 = new Sphere(
		// COLOR
		vector3<double>(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SPECULAR COLOR
		vector3<double>(220.0 / 255, 40.0 / 255, 120.0 / 255),
		// REFLECTIVE COLOR
		vector3<double>(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SHININESS
		100,
		// RAY
		2,
		// POSITION X Y Z
		SPH->get_c() + cam.get_u() * 9 + cam.get_w() * 10);

	Sphere* SPH4 = new Sphere(
		// COLOR
		vector3<double>(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SPECULAR COLOR
		vector3<double>(220.0 / 255, 40.0 / 255, 120.0 / 255),
		// REFLECTIVE COLOR
		vector3<double>(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SHININESS
		100,
		// RAY
		2,
		// POSITION X Y Z
		SPH->get_c() - cam.get_u() * 9 + cam.get_w() * 10);

	Sphere* SPH5 = new Sphere(
		// COLOR
		vector3<double>(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SPECULAR COLOR
		vector3<double>(220.0 / 255, 40.0 / 255, 120.0 / 255),
		// REFLECTIVE COLOR
		vector3<double>(200.0 / 255, 20.0 / 255, 100.0 / 255),
		// SHININESS
		100,
		// RAY
		2,
		// POSITION X Y Z
		SPH->get_c() - cam.get_v() * 9 + cam.get_w() * 10);

	Sphere* SPH6 = new Sphere(
		// COLOR
		vector3<double>(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SPECULAR COLOR
		vector3<double>(30.0 / 255, 170.0 / 255, 80.0 / 255),
		// REFLECTIVE COLOR
		vector3<double>(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SHININESS
		10,
		// RAY
		2,
		// POSITION X Y Z
		SPH->get_c() + cam.get_v() * 9 / sqrt(2) - cam.get_u() * 9 / sqrt(2) + cam.get_w() * 50);

	Sphere* SPH7 = new Sphere(
		// COLOR
		vector3<double>(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SPECULAR COLOR
		vector3<double>(30.0 / 255, 170.0 / 255, 80.0 / 255),
		// REFLECTIVE COLOR
		vector3<double>(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SHININESS
		10,
		// RAY
		2,
		// POSITION X Y Z
		SPH->get_c() + cam.get_v() * 9 / sqrt(2) + cam.get_u() * 9 / sqrt(2) + cam.get_w() * 50);

	Sphere* SPH8 = new Sphere(
		// COLOR
		vector3<double>(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SPECULAR COLOR
		vector3<double>(30.0 / 255, 170.0 / 255, 80.0 / 255),
		// REFLECTIVE COLOR
		vector3<double>(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SHININESS
		10,
		// RAY
		2,
		// POSITION X Y Z
		SPH->get_c() - cam.get_v() * 9 / sqrt(2) + cam.get_u() * 9 / sqrt(2) + cam.get_w() * 50);

	Sphere* SPH9 = new Sphere(
		// COLOR
		vector3<double>(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SPECULAR COLOR
		vector3<double>(30.0 / 255, 170.0 / 255, 80.0 / 255),
		// REFLECTIVE COLOR
		vector3<double>(10.0 / 255, 100.0 / 255, 30.0 / 255),
		// SHININESS
		10,
		// RAY
		2,
		// POSITION X Y Z
		SPH->get_c() - cam.get_v() * 9 / sqrt(2) - cam.get_u() * 9 / sqrt(2) + cam.get_w() * 50);







	// TRIANGLES # TRIANGLES # TRIANGLES # TRIANGLES # TRIANGLES
	Surface* TR1 = new Triangle(
		// COLOR
		vector3<double>(0, 0, 0),
		// SPECULAR COLOR
		vector3<double>(1, 1, 1),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// POSITION A B C
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * 20 + cam.get_v() * -10,
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * -20 + cam.get_v() * -10,
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * 0 + cam.get_v() * 20);

	Surface* TR2 = new Triangle(
		// COLOR
		vector3<double>(0, 0, 0),
		// SPECULAR COLOR
		vector3<double>(1, 1, 1),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// POSITION A B C
		cam.get_w() * 0 + cam.get_u() * -20 + cam.get_v() * 20,
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * -20 + cam.get_v() * -10,
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * 0 + cam.get_v() * 20);

	Surface* TR3 = new Triangle(
		// COLOR
		vector3<double>(0, 0, 0),
		// SPECULAR COLOR
		vector3<double>(1, 1, 1),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// POSITION A B C
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * 20 + cam.get_v() * -10,
		cam.get_w() * 0 + cam.get_u() * 20 + cam.get_v() * 20,
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * 0 + cam.get_v() * 20);

	Surface* TR4 = new Triangle(
		// COLOR
		vector3<double>(0, 0, 0),
		// SPECULAR COLOR
		vector3<double>(1, 1, 1),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// POSITION A B C
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * 20 + cam.get_v() * -10,
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * -20 + cam.get_v() * -10,
		cam.get_w() * 0 + cam.get_v() * -25);

	Surface* TR5 = new Triangle(
		// COLOR
		vector3<double>(0, 0, 0),
		// SPECULAR COLOR
		vector3<double>(1, 1, 1),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// POSITION A B C
		cam.position + cam.get_w() * 0.2 + cam.get_u() * 200 + cam.get_v() * -200,
		cam.position + cam.get_w() * 0.2 + cam.get_u() * -200 + cam.get_v() * -200,
		cam.position + cam.get_w() * 0.2 + cam.get_u() * 0 + cam.get_v() * 200);

	Surface* TR6 = new Triangle(
		// COLOR
		vector3<double>(0, 0, 0),
		// SPECULAR COLOR
		vector3<double>(1, 1, 1),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// POSITION A B C
		cam.get_w() * 0 + cam.get_u() * -20 + cam.get_v() * 20,
		cam.get_w() * 0 + cam.get_u() * 20 + cam.get_v() * 20,
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * 0 + cam.get_v() * 20);

	Surface* TR7 = new Triangle(
		// COLOR
		vector3<double>(0, 0, 0),
		// SPECULAR COLOR
		vector3<double>(1, 1, 1),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// POSITION A B C
		cam.get_w() * sqrt(2) * 40 * -1 + cam.get_u() * 20 + cam.get_v() * -10,
		cam.get_w() * 0 + cam.get_u() * 20 + cam.get_v() * 20,
		cam.get_w() * 0 + cam.get_v() * -25);

	Surface* TR8 = new Triangle(
		// COLOR
		vector3<double>(0, 0, 0),
		// SPECULAR COLOR
		vector3<double>(1, 1, 1),
		// REFLECTIVE COLOR
		vector3<double>(1, 1, 1),
		// SHININESS
		500.0,
		// POSITION A B C
		cam.get_w() * sqrt(2) * 40 * -1 - cam.get_u() * 20 + cam.get_v() * -10,
		cam.get_w() * 0 - cam.get_u() * 20 + cam.get_v() * 20,
		cam.get_w() * 0 + cam.get_v() * -25);



	Light Source1{ vector3<double>(0, 0, 30), vector3<double>(1, 1, 1) };
	Light Source2{ cam.get_u() * -19 + cam.get_v() * 19, vector3<double>(0, 0, 1) };
	Light Source3{ cam.get_u() * 19 + cam.get_v() * 19, vector3<double>(0, 1, 0) };
	Light Source4{ cam.get_v() * -24, vector3<double>(1, 0, 0) };


	scene.lights.push_back(Source1);
	scene.lights.push_back(Source2);
	scene.lights.push_back(Source3);
	scene.lights.push_back(Source4);

	scene.objs.emplace_back(TR1);
	scene.objs.emplace_back(TR2);
	scene.objs.emplace_back(TR3);
	scene.objs.emplace_back(TR4);
	scene.objs.emplace_back(TR5);
	scene.objs.emplace_back(TR6);
	scene.objs.emplace_back(TR7);
	scene.objs.emplace_back(TR8);

	//obj.emplace_back(SPH);
	scene.objs.emplace_back(SPH2);
	scene.objs.emplace_back(SPH3);
	scene.objs.emplace_back(SPH4);
	scene.objs.emplace_back(SPH5);
	scene.objs.emplace_back(SPH6);
	scene.objs.emplace_back(SPH7);
	scene.objs.emplace_back(SPH8);
	scene.objs.emplace_back(SPH9);

	//Image img("Textures/Earth.jpg", 512, 1024);
	//img.Write("Slack.png");
	
	sf::Image img;
	img.create((unsigned int)h, (unsigned int)w, sf::Color(0, 0, 0));
	std::vector<std::thread> threads;

	std::atomic<int> progress(0);
	const int threadsNum = 6;
	int lowerBound = 0;
	int upperBound = 0;
	int AntiAliasing = 7;
	for (int i = 0; i < threadsNum; i++) {
		lowerBound = upperBound;
		upperBound += img.getSize().y / threadsNum + 1;
		if (upperBound >= img.getSize().y)
			upperBound = img.getSize().y;
		threads.push_back(std::thread(line_render, std::ref(scene), &img, bounce, cam, &progress, lowerBound, upperBound, AntiAliasing));
	}
	threads.push_back(std::thread(window, std::ref(scene), &img, cam, &progress));
	while (progress < img.getSize().y * AntiAliasing * AntiAliasing) {
		std::cout << '\r' << ((double)progress.load()) / ((double)img.getSize().y * AntiAliasing * AntiAliasing) * 100;
	}

	for (auto& thread : threads)
		thread.join();
	
	img.saveToFile("file.png");
}