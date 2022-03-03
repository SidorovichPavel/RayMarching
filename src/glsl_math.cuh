#include "cuda_runtime.h"

#include "..\..\tinyAlgebra\src\glm\glm.hpp"

__device__ float sdf_union(float a, float b)
{
	return fminf(a, b);
}

__device__ glm::vec2 sdf_union_id(glm::vec2 a, glm::vec2 b)
{
	//min
	glm::vec2 res[] = { b,a };
	bool idx = a.x < b.x;
	return res[idx];
}

__device__ float sdf_intersection(float a, float b)
{
	return fmaxf(a, b);
}

__device__ glm::vec2 sdf_intersect_id(glm::vec2 a, glm::vec2 b)
{
	//max
	glm::vec2 res[] = { a,b };
	bool idx = a.x < b.x;
	return res[idx];
}

__device__ float sdf_difference(float a, float b)
{
	return fmaxf(-a, b);
}

__device__ glm::vec2 sdf_diff_id(glm::vec2 a, glm::vec2 b)
{
	glm::vec2 res[] = { -a, b };
	bool idx = -a.x < b.x;
	return res[idx];
}

__device__ float sdf_plane(glm::vec3 p, glm::vec3 n, float h)
{
	return dot(p, n) + h;
}

__device__ float sdf_sphere(glm::vec3 p, float r)
{
	return length(p) - r;
}

__device__ float sdf_box(glm::vec3 p, glm::vec3 size)
{
	glm::vec3 q = abs(p) - size;
	return length(max(q, 0.f)) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.f);
}