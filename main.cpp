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

#include "hnsw.h" 

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
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

    std::cerr << "Start building HNSW index..." << std::endl;
    
    hnsw::HNSW* appr_alg = new hnsw::HNSW(base, base_number, vecdim, M, efConstruction);

    for(size_t i = 0; i < base_number; ++i) {
        appr_alg->addPoint(i);
        
        // 打印进度条
        if ((i + 1) % 10000 == 0) {
            std::cerr << "Inserted " << (i + 1) << " / " << base_number << " points" << std::endl;
        }
    }

    std::cerr << "HNSW index build complete." << std::endl;
    return appr_alg;
}

// 获取当前时间戳字符串，用于生成文件名
std::string getCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_c);
    
    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y%m%d_%H%M%S"); // 格式：年月日_时分秒
    return oss.str();
}

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "./anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    test_number = 200;
    const size_t k = 10;
    const int efSearch = 150; 

    // 构建图索引
    hnsw::HNSW* appr_alg = build_index(base, base_number, vecdim);

    std::vector<SearchResult> results;
    results.resize(test_number);

    std::cerr << "Start testing KNN search..." << std::endl;
    
    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        auto res = appr_alg->searchKnn(test_query + i * vecdim, k, efSearch);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i * test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / k;

        results[i] = {recall, diff};
    }

    // 计算平均值
    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }
    
    avg_recall /= test_number;
    avg_latency /= test_number;

    // 终端输出
    std::cout << "average recall: " << avg_recall << "\n";
    std::cout << "average latency (us): " << avg_latency << "\n";
    
    std::string time_str = getCurrentTimeStr();
    std::string filename = "files/HNSW_linear_array" + time_str + ".txt";
    
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        outfile << "--- HNSW Serial Test Results ---\n";
        outfile << "Test Time      : " << time_str << "\n";
        outfile << "Total Queries  : " << test_number << "\n";
        outfile << "K (Top-K)      : " << k << "\n";
        outfile << "efConstruction : " << 150 << "\n";
        outfile << "efSearch       : " << efSearch << "\n";
        outfile << "M              : " << 16 << "\n";
        outfile << "--------------------------------\n";
        outfile << "Average Recall : " << std::fixed << std::setprecision(4) << avg_recall << "\n";
        outfile << "Avg Latency(us): " << std::fixed << std::setprecision(2) << avg_latency << "\n";
        outfile.close();
        std::cerr << "\n[SUCCESS] Results successfully saved to: " << filename << std::endl;
    } else {
        std::cerr << "\n[ERROR] Failed to open " << filename << " for writing! Make sure 'files/' directory exists." << std::endl;
    }

    // 释放堆内存
    delete appr_alg;
    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}