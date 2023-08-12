#ifndef OPTIX_MY_PATH_TRACER_H
#define OPTIX_MY_PATH_TRACER_H

#include <optix.h>
#include <vector_types.h>

#include <cstdint>

struct Params
{
	float4* accum_buffer;
	uchar4* frame_buffer;

	unsigned int width;
	unsigned int height;

	int subframe_index;

	float3 eye;
	float3 u;
	float3 v;
	float3 w;

	OptixTraversableHandle handle;
};

struct RayGenData { };

struct MissData
{
	float3 background_color;
};

struct LambertianData
{
	float3 albedo;
	float3 emitted;
};

struct MetalData
{
	float3 albedo;
	float3 emitted;
	float roughness;
};

struct GlassData
{
	float3 albedo;
	float3 emitted;
	float index_of_refraction;
};

struct TestData
{
	float3 albedo;
	float3 base_emitted;
};

struct HitGroupData
{
	inline constexpr HitGroupData() :
		HitGroupData(LambertianData{ })
	{ }

	inline constexpr HitGroupData(LambertianData lambertian) :
		material_index(0),
		lambertian(lambertian)
	{ }

	inline constexpr HitGroupData(MetalData metal) :
		material_index(1),
		metal(metal)
	{ }

	inline constexpr HitGroupData(GlassData glass) :
		material_index(2),
		glass(glass)
	{ }

	inline constexpr HitGroupData(TestData test) :
		material_index(3),
		test(test)
	{ }

	unsigned int material_index;

	union
	{
		LambertianData lambertian;
		MetalData metal;
		GlassData glass;
		TestData test;
	};
};

struct CallableData
{

};

struct Vertex
{
	inline constexpr Vertex(float x, float y, float z) :
		x(x), y(y), z(z), pad(0)
	{ }

	inline constexpr Vertex(float3 v) :
		Vertex(v.x, v.y, v.z)
	{ }

	inline constexpr operator float3() const
	{
		return { x, y, z };
	}

	float x, y, z, pad;
};

struct Index
{
	inline constexpr Index(uint32_t a, uint32_t b, uint32_t c) :
		a(a), b(b), c(c), pad(0)
	{ }

	inline constexpr Index(uint3 i) :
		Index(i.x, i.y, i.z)
	{ }

	inline constexpr operator uint3() const
	{
		return { a, b, c };
	}

	uint32_t a, b, c, pad;
};

#endif//OPTIX_MY_PATH_TRACER_H