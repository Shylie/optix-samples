#pragma once

//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <cstdint>
#include <vector_types.h>
#include <vector_functions.h>

constexpr unsigned int RAY_TYPE_COUNT = 1;

constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE  = OPTIX_PAYLOAD_TYPE_ID_0;

struct RadiancePRD
{
	float3       attenuation;
	unsigned int seed;
	float3       emitted;
	float        distance;
	float3       origin;
	float3       direction;
	int          done;
};

const unsigned int radiancePayloadSemantics[15] =
{
	// RadiancePRD::attenuation
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
	// RadiancePRD::seed
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
	// RadiancePRD::emitted
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
	// RadiancePRD::distance
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE ,
	// RadiancePRD::origin
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
	// RadiancePRD::direction
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
	// RadiancePRD::done
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
};

struct Params
{
	unsigned int subframe_index;
	float4*      accum_buffer;
	uchar4*      frame_buffer;
	unsigned int width;
	unsigned int height;
	unsigned int samples_per_launch;

	float3       eye;
	float3       U;
	float3       V;
	float3       W;

	OptixTraversableHandle handle;

	float distance_scale;

	float4* normals;
	uint4* indices;
};

struct RayGenData
{
};

struct MissData
{
	float4 bg_color;
};

struct HitGroupData
{
	float3 emission_color;
	float3 diffuse_color;
};

struct Vertex
{
	float x, y, z, pad;

	inline Vertex(float x, float y, float z) :
		x(x),
		y(y),
		z(z),
		pad(1)
	{ }

	inline Vertex(float3 v) :
		Vertex(v.x, v.y, v.z)
	{ }

	inline Vertex(float4 v) :
		Vertex(v.x, v.y, v.z)
	{ }

	inline operator float3() const { return make_float3(x, y, z); }
	explicit inline operator float4() const { return make_float4(x, y, z, pad); }
};

struct IndexedTriangle
{
	uint32_t v1, v2, v3, pad;

	inline IndexedTriangle(uint32_t v1, uint32_t v2, uint32_t v3) :
		v1(v1),
		v2(v2),
		v3(v3),
		pad(1)
	{ }

	inline IndexedTriangle(uint3 v) :
		IndexedTriangle(v.x, v.y, v.z)
	{ }

	inline operator uint3() const { return make_uint3(v1, v2, v3); }
};