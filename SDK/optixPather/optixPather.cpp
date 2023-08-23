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

#include <glad/glad.h>  // Needs to be included before gl_interop

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "optixPather.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "geometry.h"

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

int32_t samples_per_launch = 2;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;
typedef Record<CallableData> CallableRecord;

struct Instance
{
	float transform[12];
};

struct PathTracerState
{
	OptixDeviceContext context = 0;

	OptixTraversableHandle         gas_handle = 0;  // Traversable handle for triangle AS
	CUdeviceptr                    d_gas_output_buffer = 0;  // Triangle AS memory
	CUdeviceptr                    normals = 0;
	CUdeviceptr                    indices = 0;

	OptixModule                    ptx_module = 0;
	OptixPipelineCompileOptions    pipeline_compile_options = {};
	OptixPipeline                  pipeline = 0;

	OptixProgramGroup              raygen_prog_group = 0;
	OptixProgramGroup              miss_prog_group = 0;
	OptixProgramGroup              hit_prog_group = 0;
	OptixProgramGroup              callable_prog_groups[Material::Miss] = { };

	CUstream                       stream = 0;
	Params                         params;
	Params*                        d_params;

	OptixShaderBindingTable        sbt = {};
};


//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

const geo::GeometryData geo_data = geo::GeometryData::MakeData(
	{
		geo::Obj(std::ifstream("orb.obj"), sutil::Matrix4x4::identity(), LambertianData{ { 0.0f, 0.0f, 0.0f }, { 0.5f, 0.75f, 1.25f } }),
		geo::Obj(std::ifstream("wizard.obj"), sutil::Matrix4x4::identity(), LambertianData{ { 0.9f, 0.9f, 0.9f }, { 0.0f, 0.0f, 0.0f } })
	}
);

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (action == GLFW_PRESS)
	{
		mouse_button = button;
		trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
	}
	else
	{
		mouse_button = -1;
	}
}


static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

	if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
	{
		trackball.setViewMode(sutil::Trackball::LookAtFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
		camera_changed = true;
	}
	else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
	{
		trackball.setViewMode(sutil::Trackball::EyeFixed);
		trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
		camera_changed = true;
	}
}


static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
	// Keep rendering at the current resolution when the window is minimized.
	if (minimized)
		return;

	// Output dimensions must be at least 1 in both x and y.
	sutil::ensureMinimumSize(res_x, res_y);

	Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));
	params->width = res_x;
	params->height = res_y;
	camera_changed = true;
	resize_dirty = true;
}


static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
	minimized = (iconified > 0);
}


static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, true);
		}
	}
	else if (key == GLFW_KEY_G)
	{
		// toggle UI draw
	}
}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
	if (trackball.wheelEvent((int)yscroll))
		camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
	std::cerr << "Usage  : " << argv0 << " [options]\n";
	std::cerr << "Options: --file | -f <filename>      File for image output\n";
	std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
	std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
	std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
	std::cerr << "         --help | -h                 Print this usage message\n";
	exit(0);
}


void initLaunchParams(PathTracerState& state)
{
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&state.params.accum_buffer),
		state.params.width * state.params.height * sizeof(float4)
	));
	state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

	state.params.samples_per_launch = samples_per_launch;
	state.params.subframe_index = 0u;

	state.params.handle = state.gas_handle;

	state.params.distance_scale = 1.0f / 200.0f;

	state.params.normals = reinterpret_cast<float4*>(state.normals);
	state.params.indices = reinterpret_cast<uint4*>(state.indices);

	CUDA_CHECK(cudaStreamCreate(&state.stream));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
}


void handleCameraUpdate(Params& params)
{
	if (!camera_changed)
		return;
	camera_changed = false;

	camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
	params.eye = camera.eye();
	camera.UVWFrame(params.U, params.V, params.W);
}


void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
	if (!resize_dirty)
		return;
	resize_dirty = false;

	output_buffer.resize(params.width, params.height);

	// Realloc accumulation buffer
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&params.accum_buffer),
		params.width * params.height * sizeof(float4)
	));
}


void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
	// Update params on device
	if (camera_changed || resize_dirty)
		params.subframe_index = 0;

	handleCameraUpdate(params);
	handleResize(output_buffer, params);
}


void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state)
{
	// Launch
	uchar4* result_buffer_data = output_buffer.map();
	state.params.frame_buffer = result_buffer_data;
	CUDA_CHECK(cudaMemcpyAsync(
		reinterpret_cast<void*>(state.d_params),
		&state.params, sizeof(Params),
		cudaMemcpyHostToDevice, state.stream
	));

	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		reinterpret_cast<CUdeviceptr>(state.d_params),
		sizeof(Params),
		&state.sbt,
		state.params.width,   // launch width
		state.params.height,  // launch height
		1                     // launch depth
	));
	output_buffer.unmap();
	CUDA_SYNC_CHECK();
}


void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
	// Display
	int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
	int framebuf_res_y = 0;  //
	glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
	gl_display.display(
		output_buffer.width(),
		output_buffer.height(),
		framebuf_res_x,
		framebuf_res_y,
		output_buffer.getPBO()
	);
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void initCameraState()
{
	camera.setEye(make_float3(50.0f, 38.0f, 76.0f));
	camera.setLookat(make_float3(10.0f, 28.0f, 26.0f));
	camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
	camera.setFovY(45.0f);
	camera_changed = true;

	trackball.setCamera(&camera);
	trackball.setMoveSpeed(10.0f);
	trackball.setReferenceFrame(
		make_float3(1.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 1.0f),
		make_float3(0.0f, 1.0f, 0.0f)
	);
	trackball.setGimbalLock(true);
}

void createContext(PathTracerState& state)
{
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	OptixDeviceContext context;
	CUcontext          cu_ctx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
#ifdef DEBUG
	// This may incur significant performance cost and should only be done during development.
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
	OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

	state.context = context;
}


void buildMeshAccel(PathTracerState& state)
{
	//
	// copy mesh data to device
	//
	CUdeviceptr d_vertices = 0;
	const size_t vertices_size_in_bytes = geo_data.vertices.size() * sizeof(Vertex);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_vertices),
		geo_data.vertices.data(),
		vertices_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	const size_t vertex_indices_size_in_bytes = geo_data.vertex_indices.size() * sizeof(IndexedTriangle);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.indices), vertex_indices_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(state.indices),
		geo_data.vertex_indices.data(),
		vertex_indices_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	CUdeviceptr  d_mat_indices = 0;
	const size_t mat_indices_size_in_bytes = geo_data.material_indices.size() * sizeof(uint32_t);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_mat_indices),
		geo_data.material_indices.data(),
		mat_indices_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	const size_t normals_size_in_bytes = geo_data.normals.size() * sizeof(Vertex);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.normals), normals_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(state.normals),
		geo_data.normals.data(),
		normals_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	//
	// Build triangle GAS
	//
	std::vector<uint32_t> triangle_input_flags(geo_data.materials.size(), OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

	OptixBuildInput triangle_input = {};
	triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
	triangle_input.triangleArray.numVertices = static_cast<uint32_t>(geo_data.vertices.size());
	triangle_input.triangleArray.vertexBuffers = &d_vertices;

	triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangle_input.triangleArray.indexStrideInBytes = sizeof(IndexedTriangle);
	triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(geo_data.vertex_indices.size());
	triangle_input.triangleArray.indexBuffer = state.indices;

	triangle_input.triangleArray.numSbtRecords = geo_data.materials.size();
	triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
	triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
	triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

	triangle_input.triangleArray.flags = triangle_input_flags.data();

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes gas_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		state.context,
		&accel_options,
		&triangle_input,
		1,  // num_build_inputs
		&gas_buffer_sizes
	));

	CUdeviceptr d_temp_buffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

	// non-compacted output
	CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
	size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
		compactedSizeOffset + 8
	));

	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

	OPTIX_CHECK(optixAccelBuild(
		state.context,
		0,                                  // CUDA stream
		&accel_options,
		&triangle_input,
		1,                                  // num build inputs
		d_temp_buffer,
		gas_buffer_sizes.tempSizeInBytes,
		d_buffer_temp_output_gas_and_compacted_size,
		gas_buffer_sizes.outputSizeInBytes,
		&state.gas_handle,
		&emitProperty,                      // emitted property list
		1                                   // num emitted properties
	));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

	size_t compacted_gas_size;
	CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

	if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));

		// use handle as input and output
		OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle));

		CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
	}
	else
	{
		state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
	}
}

void createModule(PathTracerState& state)
{
	OptixPayloadType payloadType = {};
	// radiance prd
	payloadType.numPayloadValues = sizeof(radiancePayloadSemantics) / sizeof(radiancePayloadSemantics[0]);
	payloadType.payloadSemantics = radiancePayloadSemantics;

	OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
	module_compile_options.numPayloadTypes = 1;
	module_compile_options.payloadTypes = &payloadType;

	state.pipeline_compile_options.usesMotionBlur = false;
	state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	state.pipeline_compile_options.numPayloadValues = 0;
	state.pipeline_compile_options.numAttributeValues = 2;
	state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

	size_t      inputSize = 0;
	const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixPather.cu", inputSize);

	OPTIX_CHECK_LOG(optixModuleCreate(
		state.context,
		&module_compile_options,
		&state.pipeline_compile_options,
		input,
		inputSize,
		LOG, &LOG_SIZE,
		&state.ptx_module
	));
}


void createProgramGroups(PathTracerState& state)
{
	OptixProgramGroupOptions  program_group_options = {};

	{
		OptixProgramGroupDesc raygen_prog_group_desc = {};
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = state.ptx_module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			state.context, &raygen_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			LOG, &LOG_SIZE,
			&state.raygen_prog_group
		));
	}

	{
		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = state.ptx_module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			state.context, &miss_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			LOG, &LOG_SIZE,
			&state.miss_prog_group
		));
	}

	{
		OptixProgramGroupDesc hit_prog_group_desc = {};
		hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
		hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			state.context,
			&hit_prog_group_desc,
			1,  // num program groups
			&program_group_options,
			LOG, &LOG_SIZE,
			&state.hit_prog_group
		));
	}
	
	{
		OptixProgramGroupDesc callable_prog_group_descs[Material::Miss] = { };

		callable_prog_group_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		callable_prog_group_descs[0].callables.moduleDC = state.ptx_module;
		callable_prog_group_descs[0].callables.entryFunctionNameDC = "__direct_callable__lambertian";

		callable_prog_group_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		callable_prog_group_descs[1].callables.moduleDC = state.ptx_module;
		callable_prog_group_descs[1].callables.entryFunctionNameDC = "__direct_callable__metal";

		callable_prog_group_descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		callable_prog_group_descs[2].callables.moduleDC = state.ptx_module;
		callable_prog_group_descs[2].callables.entryFunctionNameDC = "__direct_callable__glass";

		callable_prog_group_descs[3].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
		callable_prog_group_descs[3].callables.moduleDC = state.ptx_module;
		callable_prog_group_descs[3].callables.entryFunctionNameDC = "__direct_callable__test";

		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			state.context,
			callable_prog_group_descs,
			Material::Miss,
			&program_group_options,
			LOG, &LOG_SIZE,
			state.callable_prog_groups
		));
	}
}


void createPipeline(PathTracerState& state)
{
	OptixProgramGroup program_groups[] =
	{
		state.raygen_prog_group,
		state.miss_prog_group,
		state.hit_prog_group,
	};

	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = 1;

	OPTIX_CHECK_LOG(optixPipelineCreate(
		state.context,
		&state.pipeline_compile_options,
		&pipeline_link_options,
		program_groups,
		sizeof(program_groups) / sizeof(program_groups[0]),
		LOG, &LOG_SIZE,
		&state.pipeline
	));

	// We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
	// parameters to optixPipelineSetStackSize.
	OptixStackSizes stack_sizes = {};
	OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prog_group, &stack_sizes, state.pipeline));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(state.miss_prog_group, &stack_sizes, state.pipeline));
	OPTIX_CHECK(optixUtilAccumulateStackSizes(state.hit_prog_group, &stack_sizes, state.pipeline));

	for (int i = 0; i < Material::Miss; i++)
	{
		OPTIX_CHECK(optixUtilAccumulateStackSizes(state.callable_prog_groups[i], &stack_sizes, state.pipeline));
	}

	uint32_t max_trace_depth = 2;
	uint32_t max_cc_depth = 0;
	uint32_t max_dc_depth = 0;
	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(
		&stack_sizes,
		max_trace_depth,
		max_cc_depth,
		max_dc_depth,
		&direct_callable_stack_size_from_traversal,
		&direct_callable_stack_size_from_state,
		&continuation_stack_size
	));

	const uint32_t max_traversal_depth = 1;
	OPTIX_CHECK(optixPipelineSetStackSize(
		state.pipeline,
		direct_callable_stack_size_from_traversal,
		direct_callable_stack_size_from_state,
		continuation_stack_size,
		max_traversal_depth
	));
}


void createSBT(PathTracerState& state)
{
	CUdeviceptr  d_raygen_record;
	const size_t raygen_record_size = sizeof(RayGenRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

	RayGenRecord rg_sbt = {};
	OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_raygen_record),
		&rg_sbt,
		raygen_record_size,
		cudaMemcpyHostToDevice
	));


	CUdeviceptr  d_miss_records;
	const size_t miss_record_size = sizeof(MissRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

	MissRecord ms_sbt[1];
	OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt[0]));
	ms_sbt[0].data.bg_color = make_float4(0.0f);

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_miss_records),
		ms_sbt,
		miss_record_size * RAY_TYPE_COUNT,
		cudaMemcpyHostToDevice
	));

	CUdeviceptr  d_hitgroup_records;
	const size_t hitgroup_record_size = sizeof(HitGroupRecord);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_hitgroup_records),
		hitgroup_record_size * RAY_TYPE_COUNT * geo_data.materials.size()
	));

	std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT * geo_data.materials.size());
	for (int i = 0; i < geo_data.materials.size(); ++i)
	{
		{
			const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

			OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_prog_group, &hitgroup_records[sbt_idx]));
			hitgroup_records[sbt_idx].data = geo_data.materials[i];
		}

		// Note that we do not need to use any program groups for occlusion
		// rays as they are traced as 'probe rays' with no shading.
	}

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_hitgroup_records),
		hitgroup_records.data(),
		hitgroup_record_size * RAY_TYPE_COUNT * geo_data.materials.size(),
		cudaMemcpyHostToDevice
	));

	CallableRecord callable_records[Material::Miss];
	for (int i = 0; i < Material::Miss; i++)
	{
		OPTIX_CHECK(optixSbtRecordPackHeader(state.callable_prog_groups[i], &callable_records[i]));
	}

	CUdeviceptr d_callable_records;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_callable_records), Material::Miss * sizeof(CallableRecord)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_callable_records),
		callable_records,
		Material::Miss * sizeof(CallableRecord),
		cudaMemcpyHostToDevice
	));

	state.sbt.raygenRecord = d_raygen_record;
	state.sbt.missRecordBase = d_miss_records;
	state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
	state.sbt.missRecordCount = RAY_TYPE_COUNT;
	state.sbt.hitgroupRecordBase = d_hitgroup_records;
	state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
	state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT * geo_data.materials.size();
	state.sbt.callablesRecordBase = d_callable_records;
	state.sbt.callablesRecordCount = Material::Miss;
	state.sbt.callablesRecordStrideInBytes = sizeof(CallableRecord);
}


void cleanupState(PathTracerState& state)
{
	OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(state.hit_prog_group));
	for (int i = 0; i < Material::Miss; i++)
	{
		OPTIX_CHECK(optixProgramGroupDestroy(state.callable_prog_groups[i]));
	}
	OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
	OPTIX_CHECK(optixDeviceContextDestroy(state.context));


	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.callablesRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.normals)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.indices)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	PathTracerState state;
	state.params.width = 768;
	state.params.height = 768;
	sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

	//
	// Parse command line options
	//
	std::string outfile;

	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--help" || arg == "-h")
		{
			printUsageAndExit(argv[0]);
		}
		else if (arg == "--no-gl-interop")
		{
			output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
		}
		else if (arg == "--file" || arg == "-f")
		{
			if (i >= argc - 1)
				printUsageAndExit(argv[0]);
			outfile = argv[++i];
		}
		else if (arg.substr(0, 6) == "--dim=")
		{
			const std::string dims_arg = arg.substr(6);
			int w, h;
			sutil::parseDimensions(dims_arg.c_str(), w, h);
			state.params.width = w;
			state.params.height = h;
		}
		else if (arg == "--launch-samples" || arg == "-s")
		{
			if (i >= argc - 1)
				printUsageAndExit(argv[0]);
			samples_per_launch = atoi(argv[++i]);
		}
		else
		{
			std::cerr << "Unknown option '" << argv[i] << "'\n";
			printUsageAndExit(argv[0]);
		}
	}

	try
	{
		initCameraState();

		//
		// Set up OptiX state
		//
		createContext(state);
		buildMeshAccel(state);
		createModule(state);
		createProgramGroups(state);
		createPipeline(state);
		createSBT(state);
		initLaunchParams(state);


		if (outfile.empty())
		{
			GLFWwindow* window = sutil::initUI("optixPathTracer", state.params.width, state.params.height);
			glfwSetMouseButtonCallback(window, mouseButtonCallback);
			glfwSetCursorPosCallback(window, cursorPosCallback);
			glfwSetWindowSizeCallback(window, windowSizeCallback);
			glfwSetWindowIconifyCallback(window, windowIconifyCallback);
			glfwSetKeyCallback(window, keyCallback);
			glfwSetScrollCallback(window, scrollCallback);
			glfwSetWindowUserPointer(window, &state.params);

			//
			// Render loop
			//
			{
				sutil::CUDAOutputBuffer<uchar4> output_buffer(
					output_buffer_type,
					state.params.width,
					state.params.height
				);

				output_buffer.setStream(state.stream);
				sutil::GLDisplay gl_display;

				std::chrono::duration<double> state_update_time(0.0);
				std::chrono::duration<double> render_time(0.0);
				std::chrono::duration<double> display_time(0.0);

				do
				{
					auto t0 = std::chrono::steady_clock::now();
					glfwPollEvents();

					updateState(output_buffer, state.params);
					auto t1 = std::chrono::steady_clock::now();
					state_update_time += t1 - t0;
					t0 = t1;

					launchSubframe(output_buffer, state);
					t1 = std::chrono::steady_clock::now();
					render_time += t1 - t0;
					t0 = t1;

					displaySubframe(output_buffer, gl_display, window);
					t1 = std::chrono::steady_clock::now();
					display_time += t1 - t0;

					sutil::displayStats(state_update_time, render_time, display_time);

					glfwSwapBuffers(window);

					++state.params.subframe_index;
				} while (!glfwWindowShouldClose(window));
				CUDA_SYNC_CHECK();
			}

			sutil::cleanupUI(window);
		}
		else
		{
			if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
			{
				sutil::initGLFW();  // For GL context
				sutil::initGL();
			}

			{
				// this scope is for output_buffer, to ensure the destructor is called bfore glfwTerminate()

				sutil::CUDAOutputBuffer<uchar4> output_buffer(
					output_buffer_type,
					state.params.width,
					state.params.height
				);

				handleCameraUpdate(state.params);
				handleResize(output_buffer, state.params);
				launchSubframe(output_buffer, state);

				sutil::ImageBuffer buffer;
				buffer.data = output_buffer.getHostPointer();
				buffer.width = output_buffer.width();
				buffer.height = output_buffer.height();
				buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

				sutil::saveImage(outfile.c_str(), buffer, false);
			}

			if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
			{
				glfwTerminate();
			}
		}

		cleanupState(state);
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		return 1;
	}

	return 0;
}
