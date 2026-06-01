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

    std::cout << "\n=== Starting HNSW OpenMP Scaling Test ===\n";

    // 设置需要采样的线程数列表
    std::vector<int> thread_configs = {1, 2, 4, 8};

    // 提前创建并打开输出文件，记录所有的测试结果
    std::string time_str = getCurrentTimeStr();
    std::string filename = "files/HNSW_scaling_results" + time_str + ".txt"; 
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "[ERROR] Failed to open results file: " << filename << std::endl;
        return -1;
    }

    // 写入文件表头信息
    outfile << "--- HNSW Multi-thread Scaling Test Results ---\n";
    outfile << "Test Time            : " << time_str << "\n";
    outfile << "Total Queries        : " << test_number << "\n";
    outfile << "K (Top-K)            : " << k << "\n";
    outfile << "efConstruction       : " << 150 << "\n";
    outfile << "efSearch             : " << efSearch << "\n";
    outfile << "M                    : " << 16 << "\n";
    outfile << "---------------------------------------------------------------------------------\n";
    outfile << std::left << std::setw(10) << "Threads" 
            << std::setw(18) << "Build_Time(ms)" 
            << std::setw(18) << "Query_Time(ms)" 
            << std::setw(15) << "Throughput(QPS)" 
            << std::setw(18) << "Avg_Latency(us)" 
            << "Recall\n";
    outfile << "---------------------------------------------------------------------------------\n";

    // 开始遍历不同的线程数量配置
    for (int current_threads : thread_configs) {
        
        // 强制设定当前 OpenMP 使用的线程数
        omp_set_num_threads(current_threads);
        
        std::cout << "\n========================================\n";
        std::cout << "Testing with Threads: " << current_threads << "\n";
        std::cout << "========================================\n";

        // 1. 构建图索引
        std::cout << "Start building HNSW index...\n";
        auto build_start_time = std::chrono::high_resolution_clock::now();
        hnsw::HNSW* index = build_index(base, base_number, vecdim);
        auto build_end_time = std::chrono::high_resolution_clock::now();
        double build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end_time - build_start_time).count();

        // 2. 初始化查询用的 VisitedList 对象池
        std::vector<hnsw::VisitedList*> query_vl_pool(current_threads);
        for (int i = 0; i < current_threads; ++i) {
            query_vl_pool[i] = new hnsw::VisitedList(base_number);
        }

        std::vector<SearchResult> results(test_number);

        // 3. 执行多线程并发查询
        std::cout << "Executing queries in parallel...\n";
        auto query_start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic, 8)
        for (size_t i = 0; i < test_number; ++i) {
            int tid = omp_get_thread_num();
            hnsw::VisitedList* vl = query_vl_pool[tid];

            auto q_start = std::chrono::high_resolution_clock::now();
            auto knn_res = index->searchKnn(test_query + i * vecdim, k, efSearch, vl);
            auto q_end = std::chrono::high_resolution_clock::now();

            int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(q_end - q_start).count();

            // 计算该条查询的 Recall
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
        
        // 4. 计算综合统计指标
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

        // 终端输出当前线程数的测试结果
        std::cout << "average recall: " << avg_recall << "\n";
        std::cout << "average latency (us): " << avg_latency << "\n";
        std::cout << "throughput (QPS): " << throughput_qps << "\n";
        std::cout << "Build Time (ms): " << build_time_ms << "\n";
        std::cout << "Query Time (ms): " << query_time_ms << "\n";

        // 将当前结果作为一行记录追加到文件中
        outfile << std::left << std::setw(10) << current_threads
                << std::setw(18) << std::fixed << std::setprecision(2) << build_time_ms
                << std::setw(18) << std::fixed << std::setprecision(2) << query_time_ms
                << std::setw(15) << std::fixed << std::setprecision(2) << throughput_qps
                << std::setw(18) << std::fixed << std::setprecision(2) << avg_latency
                << std::fixed << std::setprecision(4) << avg_recall << "\n";

        // 5. 释放当前线程数对应的图索引和查询池内存
        for (int i = 0; i < current_threads; ++i) {
            delete query_vl_pool[i];
        }
        delete index;
    }

    outfile.close();
    std::cerr << "\n[SUCCESS] All parallel scaling results successfully saved to: " << filename << std::endl;

    // 释放基础数据占用的大块内存
    delete[] base;
    delete[] test_query;
    delete[] test_gt;

    return 0;
}