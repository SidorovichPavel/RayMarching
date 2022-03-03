
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <memory>
#include <vector>
#include <array>

#include "..\..\tinyGL\src\View\View.hpp"
#include "..\..\tinyGL\src\tinyGL.hpp"
#include "..\..\tinyGL\src\Texture\Texture.hpp"
#include "..\..\tinyGL\src\Mesh\Mesh.hpp"
#include "..\..\tinyGL\src\Shader\FileShader.hpp"

#include "..\..\tinyAlgebra\src\glm\glm.hpp"
#include "..\..\tinyAlgebra\src\Camera\Camera.hpp"

__global__ void kernel(const int32_t _Width, const int32_t _Height,
	float* _Colors, float _Time);

int main()
{
	constexpr int width = 1280, int height = 720;
	constexpr size_t bytes_size = width * height * 3 * sizeof(float);

	std::vector<float> colors(width * height * 3);
	for (int i = 0; i < width * height; i++)
	{
		auto idx = i * 3;
		colors[idx + 0] = 1.f;
		colors[idx + 1] = 1.f;
		colors[idx + 2] = 1.f;
	}

	float* device_colors;
	cudaError_t error = cudaMalloc(&device_colors, bytes_size);
	if (error != cudaSuccess)
		return -1;

	constexpr int thread_x = 32, thread_y = 32;
	dim3 blocks(width / thread_x + 1, height / thread_y + 1);
	dim3 threads(thread_x, thread_y);


	

	tgl::Init();

	auto style = new tgl::Style("Ray Marching", 0, 0, width, height);
	std::unique_ptr<tgl::View> window(new tgl::View(style));

	window->init_opengl();
	window->enable_opengl_context();
	window->events().size.attach(tgl::view_port);

	tgl::gl::ActiveTexture(GL_TEXTURE1);
	tgl::Texture2D scene(colors.data(), width, height);
	scene.bind();

	tgl::FileShader shader("res/glsl/shader");
	shader.use();
	shader.uniform_int("SourceTexture", 1);

	tgl::Mesh mesh;
	std::array<float, 20> vertices
	{
		-1.f,  1.f, 0.f,		1.f, 1.f,
		-1.f, -1.f, 0.f,		1.f, 0.f,
		 1.f, -1.f, 0.f,		0.f, 0.f,
		 1.f,  1.f, 0.f,		0.f, 1.f,
	};

	std::array<uint32_t, 6> indices{ 0,1,2,2,3,0 };
	mesh.set_attribut<3, 2>(vertices.size(), vertices.data(), GL_STATIC_DRAW);
	mesh.set_indices(indices.size(), indices.data());

	tgl::gl::glOrtho(-1, 1, -1, 1, 0.01f, 500.f);

	tgl::detail::FrameTimeInfo ft_info = tgl::detail::update_frame_time(std::chrono::steady_clock::now());
	for (;; window->swap_buffers())
	{
		auto state = tgl::event_pool();
		if (!window->is_open())
			break;

		tgl::clear_black();

		kernel <<<blocks, threads>>> (width, height, device_colors, ft_info.duration.count() / 1000.f);
		error = cudaMemcpy(colors.data(), device_colors, bytes_size, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
			return -2;
		scene = tgl::Texture2D(colors.data(), width, height);
		scene.bind();
		shader.use();
		mesh.draw(GL_TRIANGLES);
		scene.unbind();

		ft_info = tgl::detail::update_frame_time(ft_info.timepoint);
		window->set_title("Ray Marching. fps:" + std::to_string(1000 / ft_info.duration.count()));
	}

	return 0;
}

#include "glsl_math.cuh"

constexpr int max_steps = 258;
constexpr float epsilon = 0.001f;
constexpr float max_dist = 500.f;
constexpr float fov = 1.f;
constexpr float max_float = std::numeric_limits<float>::max();

__device__ glm::vec2 map(const glm::vec3& p)
{
	glm::vec2 scene(max_float, 0.f), obj;

	//box
	float box_dist = sdf_box(p, glm::vec3(0.7f, 0.7f, .7f));
	float box_id = 1.f;
	glm::vec2 box = glm::vec2(box_dist, box_id);

	//sphere
	float sphere_dist = sdf_sphere(p, .95f);
	float sphere_id = 2.f;
	glm::vec2 sphere(sphere_dist, sphere_id);
	//obj = sdf_diff_id(sphere, box);
	scene = sdf_union_id(sphere, scene);
	//plane
	float plane_dist = sdf_plane(p, glm::vec3(0.f, 1.f, 0.f), 1.f);
	glm::vec2 plane(plane_dist, 3.f);
	scene = sdf_union_id(scene, plane);

	return scene;
}

__device__ glm::vec3 get_normal(const glm::vec3& p)
{
	glm::vec3 ex(epsilon, 0.f, 0.f);
	glm::vec3 ey(0.f, epsilon, 0.f);
	glm::vec3 ez(0.f, 0.f, epsilon);

	glm::vec3 n = glm::vec3(map(p).x) - glm::vec3(
		map(p - ex).x,
		map(p - ey).x,
		map(p - ez).x);

	return normalize(n);
}

__device__ glm::vec3 get_light(glm::vec3& p, glm::vec3& rd, glm::vec3 color)
{
	glm::vec3 light_pos(7.f, 5.f, -10.f);
	glm::vec3 L = normalize(light_pos - p);
	glm::vec3 N = get_normal(p);
	
	glm::vec3 diffuse = color * glm::clamp(dot(L, N), 0.f, 1.f);
	return diffuse;
}

__device__ glm::vec2 march(glm::vec3& ro, glm::vec3& rd)
{
	glm::vec2 hit, object(0.f);
	for (int i = 0; i < max_steps; i++)
	{
		glm::vec3 p = ro + object.x * rd;
		hit = map(p);
		object.x += hit.x;
		object.y = hit.y;
		if (fabsf(hit.x) < epsilon || object.x > max_dist) break;
	}
	return object;
}

__device__ glm::vec3 get_color_by_id(float fid)
{
	int id = static_cast<int>(fid);
	glm::vec3 results[] = {
		glm::vec3(1.f),
		glm::vec3(0.8f, 0.4f, 0.4f),
		glm::vec3(0.6f, 0.7f, 0.1f),
		glm::vec3(0.2f, 0.4f, 0.9f),
	};
	return results[id];
}

__device__ void render(glm::vec3& color, glm::vec2& uv)
{
	glm::vec3 ro(0.f, 0.f, -3.f);
	glm::vec3 rd = normalize(glm::vec3(uv, fov));

	glm::vec2 object = march(ro, rd);

	if (object.x < max_dist)
	{
		glm::vec3 p = ro + object.x * rd;
		color += get_light(p, rd, get_color_by_id(object.y));
	}
}



__global__ void kernel(const int32_t _Width, const int32_t _Height,
	float* _Colors, float _Time)
{
	int32_t x = fmaf(blockIdx.x, blockDim.x, threadIdx.x);
	int32_t y = fmaf(blockIdx.y, blockDim.y, threadIdx.y);

	if (!(x < _Width && y < _Height))
		return;

	glm::vec2 uv = (2.f * glm::vec2(x, y) - glm::vec2(_Width, _Height)) / static_cast<float>(_Height);

	glm::vec3 color(0.f);
	render(color, uv);

	color = glm::vec3(powf(color.x, 0.435f), powf(color.y, 0.435f), powf(color.z, 0.435f));

	int32_t idx = (y * _Width + x) * 3;
	_Colors[idx + 0] = color.x;
	_Colors[idx + 1] = color.y;
	_Colors[idx + 2] = color.z;
}