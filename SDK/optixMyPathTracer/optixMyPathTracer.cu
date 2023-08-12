#include "optixMyPathTracer.h"

#include "random.h"
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

struct HitInfo
{
	float3 position;
	float3 normal;
	float3 normal_forward;
	float3 ray_direction;
};

struct MaterialReturnInfo
{
	float3 albedo;
	float3 emitted;
	float3 new_ray_direction;
};

extern "C"
{
	__constant__ Params params;
}

extern "C" __global__ void __raygen__rg()
{
	const int width = params.width;
	const int height = params.height;
	const float3 eye = params.eye;
	const float3 u = params.u;
	const float3 v = params.v;
	const float3 w = params.w;

	const uint3 idx = optixGetLaunchIndex();
	const int subframe_index = params.subframe_index;

	unsigned int seed = tea<4>(idx.y * width + idx.x, subframe_index);

	float3 result = make_float3(0.0f);

	constexpr int NUM_SAMPLES = 1;
	int samples = NUM_SAMPLES;

	do
	{
		const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

		const float2 d = 2.0f * make_float2(
			(static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(width),
			(static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(height)
		) - 1.0f;
		float3 ray_direction = normalize(d.x * u + d.y * v + w);
		float3 ray_origin = eye;

		float3 attenuation = { 1.0f, 1.0f, 1.0f };
		float totalDistance = 0.0f;

		float3 pos = ray_origin;
		for (int i = 0; i < 100000; i += 1)
		{
			pos += ray_direction / 10;

			if (length(pos - make_float3(343.0f, 548.5f, 227.0f)) < 10.0f)
			{
				printf("%d: [%f, %f, %f]\n", i, pos.x, pos.y, pos.z);
			}
		}

		bool done = false;
		for (int i = 0; i < 1 && !done; i++)
		{
			unsigned int p0;
			unsigned int p1 = seed;
			unsigned int p2, p3, p4; // attenuation
			unsigned int p5, p6, p7; // emitted color
			unsigned int p8, p9, p10; // new origin
			unsigned int p11, p12, p13; // new direction
			unsigned int p14; // tmax

			optixTrace(
				params.handle,
				ray_origin,
				ray_direction,
				0.01f,
				1e16f,
				0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				0,
				1,
				0,
				p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14
			);

			done = p0;

			seed = p1;

			float3 emitted;
			emitted.x = __uint_as_float(p5);
			emitted.y = __uint_as_float(p6);
			emitted.z = __uint_as_float(p7);

			const float distance = fmax(fabsf(__uint_as_float(p14)) / 1000.0f, 0.0f);
			totalDistance += distance;

			result += emitted * attenuation / (totalDistance * totalDistance);

			attenuation.x *= __uint_as_float(p2);
			attenuation.y *= __uint_as_float(p3);
			attenuation.z *= __uint_as_float(p4);

			ray_origin.x = __uint_as_float(p8);
			ray_origin.y = __uint_as_float(p9);
			ray_origin.z = __uint_as_float(p10);

			ray_direction.x = __uint_as_float(p11);
			ray_direction.y = __uint_as_float(p12);
			ray_direction.z = __uint_as_float(p13);
		}
	}
	while (--samples);

	result /= static_cast<float>(NUM_SAMPLES);

	const unsigned int image_index = idx.y * params.width + idx.x;
	if (subframe_index > 0)
	{
		const float a = 1.0f / static_cast<float>(subframe_index + 1);
		const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
		result = lerp(accum_color_prev, result, a);
	}

	params.accum_buffer[image_index] = make_float4(result, 1.0f);
	params.frame_buffer[image_index] = make_color(result);
}

extern "C" __global__ void __miss__ms()
{
	MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
	optixSetPayload_0(true);
	optixSetPayload_2(__float_as_uint(1.0f));
	optixSetPayload_3(__float_as_uint(1.0f));
	optixSetPayload_4(__float_as_uint(1.0f));
	optixSetPayload_5(__float_as_uint(miss_data->background_color.x));
	optixSetPayload_6(__float_as_uint(miss_data->background_color.y));
	optixSetPayload_7(__float_as_uint(miss_data->background_color.z));
}

extern "C" __global__ void __closesthit__ch()
{
	printf("hit\n");

	HitGroupData* mat = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

	optixSetPayload_0(false);

	const float tmax = optixGetRayTmax();

	optixSetPayload_14(__float_as_uint(tmax));
	
	const float3 ray_direction = optixGetWorldRayDirection();
	const float3 hit_point = optixGetWorldRayOrigin() + tmax * ray_direction;

	optixSetPayload_8(__float_as_uint(hit_point.x));
	optixSetPayload_9(__float_as_uint(hit_point.y));
	optixSetPayload_10(__float_as_uint(hit_point.z));

	float3 verts[3];
	optixGetTriangleVertexData(params.handle, optixGetPrimitiveIndex(), optixGetSbtGASIndex(), 0.0f, verts);

	const float3 normal = normalize(cross(verts[1] - verts[0], verts[2] - verts[0]));
	const float3 normal_forward = faceforward(normal, -ray_direction, normal);

	const HitInfo hit = { hit_point, normal, normal_forward, ray_direction };

	unsigned int seed = optixGetPayload_1();

	MaterialReturnInfo mri = optixDirectCall<MaterialReturnInfo, const HitInfo&, const HitGroupData&, unsigned int&>(mat->material_index, hit, *mat, seed);

	optixSetPayload_2(__float_as_uint(mri.albedo.x));
	optixSetPayload_3(__float_as_uint(mri.albedo.y));
	optixSetPayload_4(__float_as_uint(mri.albedo.z));
	optixSetPayload_5(__float_as_uint(mri.emitted.x));
	optixSetPayload_6(__float_as_uint(mri.emitted.y));
	optixSetPayload_7(__float_as_uint(mri.emitted.z));

	optixSetPayload_11(__float_as_uint(mri.new_ray_direction.x));
	optixSetPayload_12(__float_as_uint(mri.new_ray_direction.y));
	optixSetPayload_13(__float_as_uint(mri.new_ray_direction.z));

	optixSetPayload_1(seed);
}

__device__ float3 random_unit_vector(unsigned int& seed)
{
	float3 random_variation;
	do
	{
		random_variation = 2.0f * make_float3(rnd(seed), rnd(seed), rnd(seed)) - 1.0f;
	} while (length(random_variation) > 1.0f);
	return normalize(random_variation);
}

extern "C" __device__ MaterialReturnInfo __direct_callable__lambertian(const HitInfo& hit, const HitGroupData& material, unsigned int& seed)
{
	return
	{
		material.lambertian.albedo,
		material.lambertian.emitted,
		hit.normal_forward * 1.01f + random_unit_vector(seed) * 0.99f
	};
}

extern "C" __device__ MaterialReturnInfo __direct_callable__metal(const HitInfo& hit, const HitGroupData& material, unsigned int& seed)
{
	return
	{
		material.metal.albedo,
		material.metal.emitted,
		reflect(hit.ray_direction, hit.normal_forward) + random_unit_vector(seed) * material.metal.roughness
	};
}

extern "C" __device__ MaterialReturnInfo __direct_callable__glass(const HitInfo& hit, const HitGroupData& material, unsigned int& seed)
{
	MaterialReturnInfo mri;
	mri.albedo = material.glass.albedo;
	mri.emitted = material.glass.emitted;

	if (!refract(mri.new_ray_direction, hit.ray_direction, hit.normal_forward, material.glass.index_of_refraction) && rnd(seed) > fresnel_schlick(dot(hit.ray_direction, hit.normal_forward)))
	{
		mri.new_ray_direction = reflect(hit.ray_direction, hit.normal_forward);
	}

	return mri;
}

extern "C" __device__ MaterialReturnInfo __direct_callable__test(const HitInfo& hit, const HitGroupData& material, unsigned int& seed)
{
	const float multiplier = clamp(1.0f - dot(hit.normal_forward, -hit.ray_direction), 0.0f, 1.0f);

	MaterialReturnInfo mri;
	mri.albedo = material.test.albedo;
	mri.emitted = material.test.base_emitted * multiplier;
	mri.new_ray_direction = hit.normal_forward * 1.01f + random_unit_vector(seed) * 0.99f;

	return mri;
}