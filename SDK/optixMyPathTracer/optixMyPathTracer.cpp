#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include "optixMyPathTracer.h"
#include "geometry.h"

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/GLDisplay.h>

#include <GLFW/glfw3.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <string>

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RayGenSbtRecord = SbtRecord<RayGenData>;
using MissSbtRecord = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;
using CallableSbtRecord = SbtRecord<CallableData>;

static constexpr size_t MAT_TYPE_COUNT = 4;

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr d_gas_output_buffer = 0;

    OptixModule module = 0;
    OptixPipelineCompileOptions pipeline_compile_options = { };
    OptixPipeline pipeline = 0;

    OptixProgramGroup raygen_program_group = 0;
    OptixProgramGroup miss_program_group = 0;
    OptixProgramGroup hit_program_group = 0;
    OptixProgramGroup callable_program_groups[MAT_TYPE_COUNT] = { };

    CUstream stream = 0;
    Params params;
    Params* d_params;

    OptixShaderBindingTable sbt = { };
};

static bool camera_changed = true;
static sutil::Camera camera;
static sutil::Trackball trackball;

static int32_t mouse_button = -1;

static void contextLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << std::endl;
}

static void createContext(PathTracerState& state)
{
	CUDA_CHECK(cudaFree(0));

	OptixDeviceContext context;
	CUcontext cuContext = 0;
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = { };
	options.logCallbackFunction = &contextLogCallback;
	options.logCallbackLevel = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &context));

	state.context = context;
}

static geo::GeometryData data;
//= geo::GeometryData::MakeData(
//    {
//        geo::Box({ 0, 0, 0 }, { 10, 10, 10 }, LambertianData{ { 0.80f, 0.80f, 0.80f }, { 5.0f, 5.0f, 5.0f } }),
//        //geo::Icosphere({ 0, 0, 0 }, 2.5f, 0, LambertianData{ { 1.00f, 1.00f, 1.00f }, { 5.0f, 5.0f, 5.0f } }),
//    }
//);

static void buildMeshAccel(PathTracerState& state)
{
    data.materials.push_back(LambertianData{ { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f } });
    data.material_indices = geo::MaterialIndices(4, 0);

    // floor
    data.vertices.push_back({ 0.0f,   0.0f,   0.0f });
    data.vertices.push_back({ 0.0f,   0.0f, 559.2f });
    data.vertices.push_back({ 556.0f, 0.0f, 559.2f });
    data.vertices.push_back({ 556.0f, 0.0f,   0.0f });

    data.vertex_indices.push_back({ 0, 1, 2 });
    data.vertex_indices.push_back({ 0, 2, 3 });

    // ceiling
    data.vertices.push_back({ 0.0f, 548.8f, 0.0f });
    data.vertices.push_back({ 556.0f, 548.8f, 0.0f });
    data.vertices.push_back({ 556.0f, 548.8f, 559.2f });
    data.vertices.push_back({ 0.0f, 548.8f, 559.2f });

    data.vertex_indices.push_back({ 4, 5, 6 });
    data.vertex_indices.push_back({ 4, 6, 7 });

    CUdeviceptr d_vertices;
    const size_t vertices_size = data.vertices.size() * sizeof(Vertex);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertices),
        data.vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_vertex_indices;
    const size_t vertex_indices_size = data.vertex_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertex_indices), vertex_indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_vertex_indices),
        data.vertex_indices.data(),
        vertex_indices_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_mat_indices;
    const size_t mat_indices_size = data.material_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_mat_indices),
        data.material_indices.data(),
        mat_indices_size,
        cudaMemcpyHostToDevice
    ));

    std::vector<uint32_t> triangle_input_flags(data.materials.size(), OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

    OptixBuildInput triangle_input = { };
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(data.vertices.size());
    triangle_input.triangleArray.vertexBuffers = &d_vertices;

    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    triangle_input.triangleArray.numIndexTriplets = data.vertex_indices.size();
    triangle_input.triangleArray.indexBuffer = d_vertex_indices;

    triangle_input.triangleArray.flags = triangle_input_flags.data();

    triangle_input.triangleArray.numSbtRecords = data.materials.size();
    triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBuildOptions accel_options = { };
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &triangle_input,
        1,
        &gas_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compacted_size_offset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, sizeof(size_t));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compacted_size_offset + sizeof(size_t)
    ));

    OptixAccelEmitDesc emitProperty = { };
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compacted_size_offset);

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0,
        &accel_options,
        &triangle_input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty,
        1
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertex_indices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));

        OPTIX_CHECK(optixAccelCompact(
            state.context,
            0,
            state.gas_handle,
            state.d_gas_output_buffer,
            compacted_gas_size,
            &state.gas_handle
        ));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

static void createModule(PathTracerState& state)
{
    OptixModuleCompileOptions module_compile_options = { }; 
#ifndef NDEBUG
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#endif

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 15;
    state.pipeline_compile_options.numAttributeValues = 2;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER;

    size_t input_size = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixMyPathTracer.cu", input_size);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        input_size,
        LOG, &LOG_SIZE,
        &state.module
    ));
}

static void createProgramGroups(PathTracerState& state)
{
    OptixProgramGroupOptions program_group_options = { };
    
    {
        OptixProgramGroupDesc raygen_prog_group_desc = { };
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &raygen_prog_group_desc,
            1,
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.raygen_program_group
        ));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = { };
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &miss_prog_group_desc,
            1,
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.miss_program_group
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = { };
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1,
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.hit_program_group
        ));
    }

    {
        OptixProgramGroupDesc callable_prog_group_descs[MAT_TYPE_COUNT] = {};
        callable_prog_group_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_prog_group_descs[0].callables.moduleDC = state.module;
        callable_prog_group_descs[0].callables.entryFunctionNameDC = "__direct_callable__lambertian";

        callable_prog_group_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_prog_group_descs[1].callables.moduleDC = state.module;
        callable_prog_group_descs[1].callables.entryFunctionNameDC = "__direct_callable__metal";

        callable_prog_group_descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_prog_group_descs[2].callables.moduleDC = state.module;
        callable_prog_group_descs[2].callables.entryFunctionNameDC = "__direct_callable__glass";

        callable_prog_group_descs[3].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callable_prog_group_descs[3].callables.moduleDC = state.module;
        callable_prog_group_descs[3].callables.entryFunctionNameDC = "__direct_callable__test";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            callable_prog_group_descs,
            std::size(callable_prog_group_descs),
            &program_group_options,
            LOG, &LOG_SIZE,
            state.callable_program_groups
        ));
    }
}

static void createPipeline(PathTracerState& state)
{
    constexpr uint32_t MAX_TRACE_DEPTH = 1;
    constexpr uint32_t MAX_TRAVERSAL_DEPTH = 1;

    OptixProgramGroup program_groups[] =
    {
        state.raygen_program_group,
        state.miss_program_group,
        state.hit_program_group,
        state.callable_program_groups[0],
        state.callable_program_groups[1],
        state.callable_program_groups[2],
        state.callable_program_groups[3]
    };

    OptixPipelineLinkOptions pipeline_link_options = { };
    pipeline_link_options.maxTraceDepth = MAX_TRACE_DEPTH;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        std::size(program_groups),
        LOG, &LOG_SIZE,
        &state.pipeline
    ));

    OptixStackSizes stack_sizes = { };
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_program_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.miss_program_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.hit_program_group, &stack_sizes, state.pipeline));
    for (int i = 0; i < MAT_TYPE_COUNT; i++)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(state.callable_program_groups[i], &stack_sizes, state.pipeline));
    }

    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;

    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        MAX_TRACE_DEPTH,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        MAX_TRAVERSAL_DEPTH
    ));
}

static void createSBT(PathTracerState& state)
{
    CUdeviceptr d_raygen_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RayGenSbtRecord)));

    RayGenSbtRecord raygen_record = { };
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_program_group, &raygen_record));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &raygen_record,
        sizeof(RayGenSbtRecord),
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_miss_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissSbtRecord)));

    MissSbtRecord miss_record;
    miss_record.data.background_color = { 0.0f, 0.0f, 0.0f };
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_program_group, &miss_record));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_record),
        &miss_record,
        sizeof(MissSbtRecord),
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_hitgroup_records;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        data.materials.size() * sizeof(HitGroupSbtRecord)
    ));

    std::vector<HitGroupSbtRecord> hitgroup_records(data.materials.size());
    for (int i = 0; i < data.materials.size(); i++)
    {
        OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_program_group, &hitgroup_records[i]));
        hitgroup_records[i].data = data.materials[i];
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        data.materials.data(),
        data.materials.size() * sizeof(HitGroupSbtRecord),
        cudaMemcpyHostToDevice
    ));

    CallableSbtRecord callable_records[MAT_TYPE_COUNT];
    for (int i = 0; i < MAT_TYPE_COUNT; i++)
    {
        OPTIX_CHECK(optixSbtRecordPackHeader(state.callable_program_groups[i], &callable_records[i]));
    }

    CUdeviceptr d_callable_records;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_callable_records), MAT_TYPE_COUNT * sizeof(CallableSbtRecord)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_callable_records),
        callable_records,
        MAT_TYPE_COUNT * sizeof(CallableSbtRecord),
        cudaMemcpyHostToDevice
    ));

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = data.materials.size();
    state.sbt.callablesRecordBase = d_callable_records;
    state.sbt.callablesRecordCount = MAT_TYPE_COUNT;
    state.sbt.callablesRecordStrideInBytes = sizeof(CallableSbtRecord);

}

static void cleanupState(PathTracerState& state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_program_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_program_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hit_program_group));
    for (int i = 0; i < MAT_TYPE_COUNT; i++)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(state.callable_program_groups[i]));
    }
    OPTIX_CHECK(optixModuleDestroy(state.module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.callablesRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
}

static void initLaunchParams(PathTracerState& state)
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)
    ));

    state.params.frame_buffer = nullptr;

    state.params.subframe_index = 0;

    state.params.handle = state.gas_handle;

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
}

static void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state)
{
    state.params.frame_buffer = output_buffer.map();

    CUDA_CHECK(cudaMemcpyAsync(
        state.d_params,
        &state.params,
        sizeof(Params),
        cudaMemcpyHostToDevice,
        state.stream
    ));

    OPTIX_CHECK(optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(Params),
        &state.sbt,
        state.params.width,
        state.params.height,
        1
    ));

    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

static void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    int framebuf_res_x = 0;
    int framebuf_res_y = 0;
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}

static void initCameraState()
{
    camera.setEye({ 278.0f, 273.0f, -900.0f });
    camera.setLookat({ 278.0f, 273.0f, 330.0f });
    camera.setUp({ 0.0f, 1.0f, 0.0f });
    camera.setFovY(35.0f);

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f }
    );
    trackball.setGimbalLock(true);
}

static void handleCameraUpdate(Params& params)
{
    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.u, params.v, params.w);
}

static void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
    if (camera_changed)
    {
        handleCameraUpdate(params);
        camera_changed = false;

        params.subframe_index = 0;
    }
}

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

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT || mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(mouse_button == GLFW_MOUSE_BUTTON_LEFT ? sutil::Trackball::LookAtFixed : sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t, int32_t action, int32_t)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
}

static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    if (trackball.wheelEvent(yscroll))
    {
        camera_changed = true;
    }
}

int main(int argc, char** argv)
{
    PathTracerState state;
    state.params.width = 768;
    state.params.height = 768;

    try
    {
        createContext(state);
        buildMeshAccel(state);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);
        createSBT(state);
        initLaunchParams(state);

        initCameraState();

        GLFWwindow* window = sutil::initUI("optixMyPathTracer", state.params.width, state.params.height);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetWindowUserPointer(window, &state.params);

        {
            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                sutil::CUDAOutputBufferType::GL_INTEROP,
                state.params.width,
                state.params.height
            );

            output_buffer.setStream(state.stream);
            sutil::GLDisplay gl_display;

            std::chrono::duration<double> state_update_time(0);
            std::chrono::duration<double> render_time(0);
            std::chrono::duration<double> display_time(0);

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
                display_time = t1 - t0;

                sutil::displayStats(state_update_time, render_time, display_time);

                glfwSwapBuffers(window);
                
                state.params.subframe_index++;
            } while (!glfwWindowShouldClose(window));
            CUDA_SYNC_CHECK();
        }

        sutil::cleanupUI(window);

        cleanupState(state);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
}