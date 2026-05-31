#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <ctime>
#include <omp.h>

#include "hnsw.h" 

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open data file: " << data_path << std::endl;
        exit(-1);
    }
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; 
};

// 构建图索引，返回自定义的 HNSW 指针
hnsw::HNSW* build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; 
    const int M = 16; 

    std::cerr << "Start building HNSW index (Parallel mode)..." << std::endl;
    
    hnsw::HNSW* appr_alg = new hnsw::HNSW(base, base_number, vecdim, M, efConstruction);

    #pragma omp parallel for schedule(dynamic, 64)
    for(size_t i = 0; i < base_number; ++i) {
        appr_alg->addPoint(i);
        
        if (omp_get_thread_num() == 0 && (i + 1) % 5000 == 0) {
            std::cerr << "Thread 0 reporting: Processed approx " << (i + 1) << " / " << base_number << " points\r";
        }
    }

    std::cerr << "\nHNSW index build complete." << std::endl;
    return appr_alg;
}

// 获取当前系统时间的字符串函数
std::string getCurrentTimeStr() {
    std::time_t now = std::time(nullptr);
    std::tm* ltm = std::localtime(&now);
    std::ostringstream oss;
    oss << "_" << 1900 + ltm->tm_year
        << std::setw(2) << std::setfill('0') << 1 + ltm->tm_mon
        << std::setw(2) << std::setfill('0') << ltm->tm_mday
        << "_" << std::setw(2) << std::setfill('0') << ltm->tm_hour
        << std::setw(2) << std::setfill('0') << ltm->tm_min
        << std::setw(2) << std::setfill('0') << ltm->tm_sec;
    return oss.str();
}

int main() {

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<uint32_t>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 200; // 截取前 200 个用于测试
    const size_t k = 10;
    const int efSearch = 150; 

    std::cout << "\n=== Starting HNSW Parallel Optimization Test ===\n";

    auto total_start_time = std::chrono::high_resolution_clock::now();

    // 构建图索引
    std::cout << "Building HNSW index (Parallel mode)...\n";
    auto build_start_time = std::chrono::high_resolution_clock::now();
    hnsw::HNSW* index = build_index(base, base_number, vecdim);
    auto build_end_time = std::chrono::high_resolution_clock::now();
    double build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end_time - build_start_time).count();

    int max_threads = omp_get_max_threads();
    std::cout << "Initializing VisitedList pool for " << max_threads << " OpenMP threads...\n";
    std::vector<hnsw::VisitedList*> query_vl_pool(max_threads);
    for (int i = 0; i < max_threads; ++i) {
        query_vl_pool[i] = new hnsw::VisitedList(base_number);
    }

    std::vector<SearchResult> results(test_number);

    std::cout << "Executing queries in parallel using OpenMP...\n";
    auto query_start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < test_number; ++i) {
        int tid = omp_get_thread_num();
        hnsw::VisitedList* vl = query_vl_pool[tid];

        auto q_start = std::chrono::high_resolution_clock::now();
        auto knn_res = index->searchKnn(test_query + i * vecdim, k, efSearch, vl);
        auto q_end = std::chrono::high_resolution_clock::now();

        int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(q_end - q_start).count();

        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < k; ++j) {
            gt_set.insert(test_gt[i * test_gt_d + j]);
        }

        size_t match_count = 0;
        auto temp_res = knn_res;
        while (!temp_res.empty()) {
            if (gt_set.count(temp_res.top().second)) {
                match_count++;
            }
            temp_res.pop();
        }

        results[i].recall = (float)match_count / k;
        results[i].latency = latency;
    }

    auto query_end_time = std::chrono::high_resolution_clock::now();
    double query_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(query_end_time - query_start_time).count();
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time).count();

    double query_time_seconds = query_time_ms / 1000.0;
    double throughput_qps = (query_time_seconds > 0) ? (double)test_number / query_time_seconds : 0.0;

    float avg_recall = 0.0f;
    double avg_latency = 0.0;
    for (size_t i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }
    avg_recall /= test_number;
    avg_latency /= test_number;

    std::cout << "\n========================================\n";
    std::cout << "average recall: " << avg_recall << "\n";
    std::cout << "average latency (us): " << avg_latency << "\n";
    std::cout << "throughput (QPS): " << throughput_qps << "\n";
    std::cout << "Build Time (ms): " << build_time_ms << "\n";
    std::cout << "Query Time (ms): " << query_time_ms << "\n";
    std::cout << "Total Running Time (ms): " << total_time_ms << "\n";
    std::cout << "========================================\n";

    std::string time_str = getCurrentTimeStr();
    std::string filename = "files/HNSW_linear_array_parallel" + time_str + ".txt"; 
    
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        outfile << "--- HNSW Parallel Test Results ---\n";
        outfile << "Test Time            : " << time_str << "\n";
        outfile << "Available Threads    : " << max_threads << "\n";
        outfile << "Total Queries        : " << test_number << "\n";
        outfile << "K (Top-K)            : " << k << "\n";
        outfile << "efConstruction       : " << 150 << "\n";
        outfile << "efSearch             : " << efSearch << "\n";
        outfile << "M                    : " << 16 << "\n";
        outfile << "--------------------------------\n";
        outfile << "Average Recall       : " << std::fixed << std::setprecision(4) << avg_recall << "\n";
        outfile << "Avg Latency(us)      : " << std::fixed << std::setprecision(2) << avg_latency << "\n";
        outfile << "Throughput (QPS)     : " << std::fixed << std::setprecision(2) << throughput_qps << "\n";
        outfile << "--------------------------------\n";
        outfile << "Build Time (ms)      : " << build_time_ms << "\n";
        outfile << "Query Time (ms)      : " << query_time_ms << "\n";
        outfile << "Total Execution (ms) : " << total_time_ms << "\n";
        outfile.close();
        std::cerr << "\n[SUCCESS] Parallel results successfully saved to: " << filename << std::endl;
    } else {
        std::cerr << "\n[ERROR] Failed to save results to: " << filename << std::endl;
    }

    for (int i = 0; i < max_threads; ++i) {
        delete query_vl_pool[i];
    }
    delete index;
    // 修复释放内存变量名
    delete[] base;
    delete[] test_query;
    delete[] test_gt;

    return 0;
}