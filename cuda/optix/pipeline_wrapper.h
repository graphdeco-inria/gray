#include "../params.h"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#include <string>
#include <filesystem>
#else
#include <dlfcn.h>
#include <unistd.h>
#include <libgen.h>
#endif

class PipelineWrapper {
  public:
    OptixDeviceContext context;
    std::string variant;

    PipelineWrapper() {
        initOptix();
        initOptixModule();
        createProgramGroups();
        createPipeline();
        createRecords();
    }

    void launch(int width, int height, CUdeviceptr params_on_device) {
        assert(params_on_device != 0);
        OPTIX_CHECK(optixLaunch(pipeline, nullptr, params_on_device, sizeof(Params), &sbt, width, height, 1));
    }

    ~PipelineWrapper() {
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));

        OPTIX_CHECK(optixProgramGroupDestroy(raygen_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(miss_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(hit_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(ellipsoid_hit_pg));

        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        OPTIX_CHECK(optixModuleDestroy(module));

        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }

  private:
    OptixModule module = nullptr;
    OptixShaderBindingTable sbt = {};

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline;

    OptixProgramGroup raygen_pg;
    OptixProgramGroup miss_pg;
    OptixProgramGroup hit_pg;
    OptixProgramGroup ellipsoid_hit_pg;

    static void context_log_cb(uint32_t level, const char *tag, const char *message, void *) {
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
    }

#ifdef _WIN32
    static std::string getPtxPath() {
        HMODULE hModule = nullptr;

        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCSTR>(&PipelineWrapper::getPtxPath), &hModule)) {
            return "";
        }

        char path[MAX_PATH];
        DWORD size = GetModuleFileNameA(hModule, path, MAX_PATH);
        if (size == 0 || size == MAX_PATH)
            return "";

        std::filesystem::path p(path);
        std::filesystem::path parent = p.parent_path();
        std::filesystem::path parent_of_parent = parent.parent_path();

        for (const auto &candidate_root : {parent_of_parent, parent}) {
            std::filesystem::path candidate = candidate_root / "libgray.ptx";
            if (std::filesystem::exists(candidate))
                return candidate.string();
        }

        return (parent_of_parent / "libgray.ptx").string();
    }
#else
    static std::string getPtxPath() {
        Dl_info dl_info;
        dladdr((void *)getPtxPath, &dl_info);
        std::string path(dl_info.dli_fname);
        char *path_dup = strdup(path.c_str());
        std::string dir = dirname(path_dup);
        free(path_dup);
        return dir + "/libgray.ptx";
    }
#endif

    static std::string loadPtxFile() {
        std::string path = getPtxPath();
        std::ifstream file(path.c_str(), std::ios::binary);
        if (file.good()) {
            std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
            std::string str;
            str.assign(buffer.begin(), buffer.end());
            return str;
        } else {
            std::string error = "couldn't locate ptx file in path " + path;
            throw std::runtime_error(error);
        }
    }

    // * Boilerplate below

    void initOptix() {
        CUDA_CHECK(cudaFree(0));
        CUcontext cuCtx = 0;
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        const char *log_level_env = std::getenv("OPTIX_LOG_LEVEL");
        options.logCallbackLevel = log_level_env ? std::atoi(log_level_env) : 2;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }

    void initOptixModule() {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues = 4;
        pipeline_compile_options.numAttributeValues = 0;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

        std::string ptxData = loadPtxFile();
        OPTIX_CHECK_LOG(optixModuleCreate(context, &module_compile_options, &pipeline_compile_options, ptxData.c_str(),
                                          ptxData.size(), LOG, &LOG_SIZE, &module));
    }

    void createProgramGroups() {
        OptixProgramGroupOptions program_group_options = {}; // * Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        std::string raygen_name = "__raygen__rg";
        raygen_prog_group_desc.raygen.entryFunctionName = raygen_name.c_str();
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc, 1, &program_group_options, LOG,
                                                &LOG_SIZE, &raygen_pg));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = nullptr;
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc, 1, &program_group_options, LOG,
                                                &LOG_SIZE, &miss_pg));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
        hitgroup_prog_group_desc.hitgroup.moduleIS = module;
        std::string intersection_name = "__intersection__is";
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = intersection_name.c_str();
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_prog_group_desc, 1, &program_group_options, LOG,
                                                &LOG_SIZE, &hit_pg));

        // * Ellipsoid viewer hit group: dedicated IS + CH shaders (SBT index 1).
        OptixProgramGroupDesc ellipsoid_hitgroup_desc = {};
        ellipsoid_hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        ellipsoid_hitgroup_desc.hitgroup.moduleCH = module;
        std::string ellipsoid_ch_name = "__closesthit__ellipsoid";
        ellipsoid_hitgroup_desc.hitgroup.entryFunctionNameCH = ellipsoid_ch_name.c_str();
        ellipsoid_hitgroup_desc.hitgroup.moduleAH = nullptr;
        ellipsoid_hitgroup_desc.hitgroup.entryFunctionNameAH = nullptr;
        ellipsoid_hitgroup_desc.hitgroup.moduleIS = module;
        std::string ellipsoid_is_name = "__intersection__is_ellipsoid";
        ellipsoid_hitgroup_desc.hitgroup.entryFunctionNameIS = ellipsoid_is_name.c_str();
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &ellipsoid_hitgroup_desc, 1, &program_group_options, LOG,
                                                &LOG_SIZE, &ellipsoid_hit_pg));
    }

    void createPipeline() {
        const uint32_t max_trace_depth = 1;

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        OptixProgramGroup program_groups[] = {raygen_pg, miss_pg, hit_pg, ellipsoid_hit_pg};
        OptixProgramGroup *program_groups_array = reinterpret_cast<OptixProgramGroup *>(
            &program_groups); // * Safe since OptixProgramGroup typdefs a pointer type
        OPTIX_CHECK_LOG(
            optixPipelineCreate(context, &pipeline_compile_options, &pipeline_link_options, program_groups_array,
                                sizeof(program_groups_array) / sizeof(OptixProgramGroup), LOG, &LOG_SIZE, &pipeline));

        OptixStackSizes stack_sizes = {};
        OPTIX_CHECK(optixUtilAccumulateStackSizes(raygen_pg, &stack_sizes, pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(miss_pg, &stack_sizes, pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(hit_pg, &stack_sizes, pipeline));
        OPTIX_CHECK(optixUtilAccumulateStackSizes(ellipsoid_hit_pg, &stack_sizes, pipeline));

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0,
                                               &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size, 1));
    }

    template <typename T> struct SbtRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    void createRecords() {
        struct Empty {};

        SbtRecord<Empty> raygen_rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &raygen_rec));
        CUdeviceptr d_raygen_rec;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_rec), sizeof(raygen_rec)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_raygen_rec), &raygen_rec, sizeof(raygen_rec),
                              cudaMemcpyHostToDevice));
        sbt.raygenRecord = d_raygen_rec;

        SbtRecord<Empty> miss_rec = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &miss_rec));
        CUdeviceptr d_miss_rec;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_rec), sizeof(miss_rec)));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void *>(d_miss_rec), &miss_rec, sizeof(miss_rec), cudaMemcpyHostToDevice));
        sbt.missRecordBase = d_miss_rec;
        sbt.missRecordStrideInBytes = sizeof(miss_rec);
        sbt.missRecordCount = 1;

        // *Two hit group records: [0] = regular gaussian IS, [1] = ellipsoid IS+CH.
        SbtRecord<Empty> hitgroup_recs[2] = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(hit_pg, &hitgroup_recs[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(ellipsoid_hit_pg, &hitgroup_recs[1]));
        CUdeviceptr d_hitgroup_rec;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitgroup_rec), sizeof(hitgroup_recs)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitgroup_rec), &hitgroup_recs, sizeof(hitgroup_recs),
                              cudaMemcpyHostToDevice));
        sbt.hitgroupRecordBase = d_hitgroup_rec;
        sbt.hitgroupRecordStrideInBytes = sizeof(hitgroup_recs[0]);
        sbt.hitgroupRecordCount = 2;
    }
};
