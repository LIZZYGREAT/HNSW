#pragma once
#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
#include "distance.h"
#include "visited_list.h"

namespace hnsw {

class HNSW {
public:
    int M_;                 // 除第0层外，每层最大的邻居数
    int M0_;                // 第0层最大的邻居数，通常设为 2 * M_ 
    int efConstruction_;    // 建图时的候选队列大小
    double mult_;           // 计算随机层数的衰减因子，通常为 1 / ln(M)
    
    size_t node_num_;    // 节点总数
    size_t dim_;         // 向量维度
    float* data_;        // 底层向量数据的指针

    int max_level_;      // 当前整个图的最高层数
    int entry_node_;     // 最高层的入口节点 ID

    // 记录每个节点的最大层数
    int* node_max_level_; 
    
    // 线性邻接表：flat_links_[node_id] 指向一段连续的内存
    int** flat_links_;   

    // 建图的 VisitedList 池
    std::vector<VisitedList*> build_vl_pool_;
    // 线程的随机数生成器，防止多线程获取随机层数时发生状态竞争崩溃
    std::vector<std::default_random_engine> rng_pool_;
    std::uniform_real_distribution<double> uniform_distribution_;

    // 在双向连线修改 flat_links_ 时提供局部保护
    omp_lock_t* node_locks_;

    // 每次 pop 出最小距离。
    typedef std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> MinHeap;

    // 每次 pop 出最大距离。
    typedef std::priority_queue<std::pair<float, int>> MaxHeap;

public:
    HNSW(float* data, size_t node_num, size_t dim, int M, int efConstruction) 
        : data_(data), node_num_(node_num), dim_(dim), 
          M_(M), efConstruction_(efConstruction) {
        
        M0_ = M_ * 2;
        mult_ = 1.0 / log(1.0 * M_);
        max_level_ = -1; 
        entry_node_ = -1;

        uniform_distribution_ = std::uniform_real_distribution<double>(0.0, 1.0);

        node_max_level_ = new int[node_num_];
        flat_links_ = new int*[node_num_];
        for (size_t i = 0; i < node_num_; ++i) {
            flat_links_[i] = nullptr; 
        }

        // 初始化 OpenMP 
        int max_threads = omp_get_max_threads();
        build_vl_pool_.resize(max_threads);
        rng_pool_.resize(max_threads);
        
        for (int i = 0; i < max_threads; ++i) {
            build_vl_pool_[i] = new VisitedList(node_num_);
            rng_pool_[i].seed(2026 + i); 
        }

        node_locks_ = new omp_lock_t[node_num_];
        for (size_t i = 0; i < node_num_; ++i) {
            omp_init_lock(&node_locks_[i]);
        }
    }

    ~HNSW() {
        for (size_t i = 0; i < node_num_; ++i) {
            omp_destroy_lock(&node_locks_[i]);
        }
        delete[] node_locks_;

        for (auto vl : build_vl_pool_) {
            delete vl;
        }

        delete[] node_max_level_;
        for (size_t i = 0; i < node_num_; ++i) {
            if (flat_links_[i] != nullptr) {
                delete[] flat_links_[i];
            }
        }
        delete[] flat_links_;
    }

    // 生成随机层数
    int getRandomLevel(int tid) {
        double r = uniform_distribution_(rng_pool_[tid]);
        if (r < std::numeric_limits<double>::epsilon()) {   
            r = 1.0; 
        }
        return (int)(-log(r) * mult_);
    }

private:
    /**
     * @brief 内存偏移：获取指定节点在特定层级的邻居数组首地址
     * @note 返回的数组中，第 0 位是当前邻居数量，后续是邻居 ID
     */
     inline int* getList(int node_id, int level) const {
        int* node_memory = flat_links_[node_id];
        if (node_memory == nullptr) return nullptr; 
        
        if (level == 0) {
            return node_memory;
        } else {
            return node_memory + (M0_ + 2) + (level - 1) * (M_ + 2);  
        }
    }

    /** * @brief 在指定的 level 层内进行贪心搜索
     * @param vl 外部传入的、当前线程的 VisitedList 指针
     */
    MaxHeap searchLayer(const float* query, MaxHeap enter_points, int ef, int level, VisitedList* vl) {
        vl->advance();

        MinHeap candidates; 
        MaxHeap results;    

        while (!enter_points.empty()) {
            std::pair<float, int> ep = enter_points.top();
            enter_points.pop();
            
            candidates.push(ep);
            results.push(ep);
            vl->markVisited(ep.second);
        }

        // 贪心游走
        while (!candidates.empty()) {
            std::pair<float, int> curr_c = candidates.top();
            candidates.pop();

            float dist_c = curr_c.first;
            int node_c = curr_c.second;
            float dist_f = results.top().first;

            if (dist_c > dist_f) {
                break; 
            }

            int* neighbors = getList(node_c, level);
            if (neighbors == nullptr) {
                continue;
            }

            int num_neighbors = neighbors[0]; 

            for (int i = 1; i <= num_neighbors; ++i) {
                int neighbor_id = neighbors[i];

                if (!vl->isVisited(neighbor_id)) {
                    vl->markVisited(neighbor_id);

                    float dist_e = get_distance(
                        query, 
                        data_ + (size_t)neighbor_id * dim_, 
                        dim_
                    );

                    if (results.size() < (size_t)ef || dist_e < results.top().first) {
                        candidates.push({dist_e, neighbor_id});
                        results.push({dist_e, neighbor_id});

                        if (results.size() > (size_t)ef) {
                            results.pop(); 
                        }
                    }
                }
            }
        }

        return results;
    }

    // 启发式选边
    std::vector<int> getNeighbors(MaxHeap candidates, int M_max) {
        if (candidates.size() <= (size_t)M_max) {    
            std::vector<int> res;
            std::vector<std::pair<float, int>> temp;

            while (!candidates.empty()) {
                temp.push_back(candidates.top());
                candidates.pop();
            }
            std::reverse(temp.begin(), temp.end()); 
            for (auto& p : temp) res.push_back(p.second);
            return res;
        }

        std::vector<std::pair<float, int>> sorted_candidates;
        while (!candidates.empty()) {
            sorted_candidates.push_back(candidates.top());
            candidates.pop();
        }
        std::reverse(sorted_candidates.begin(), sorted_candidates.end());

        std::vector<int> result_neighbors;
        for (const auto& c : sorted_candidates) {
            if (result_neighbors.size() >= (size_t)M_max) break;

            int candidate_id = c.second;
            float dist_to_candidate = c.first;  

            bool keep = true;
            for (int result_neighbor_id : result_neighbors) {
                float dist_to_neighbor = get_distance(
                    data_ + (size_t)candidate_id * dim_,
                    data_ + (size_t)result_neighbor_id * dim_,
                    dim_
                );
                if (dist_to_neighbor < dist_to_candidate) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                result_neighbors.push_back(candidate_id);
            }
        }
        return result_neighbors;
    }

public:
    void addPoint(int new_node_id) {
        int tid = omp_get_thread_num(); 
        int insert_level = getRandomLevel(tid);
        node_max_level_[new_node_id] = insert_level;  

        // 空间预分配
        size_t total_memory_size = (M0_ + 2) + insert_level * (M_ + 2);
        flat_links_[new_node_id] = new int[total_memory_size];

        for (int l = 0; l <= insert_level; ++l) {
            getList(new_node_id, l)[0] = 0; 
        }

        // 检查是否为图中首个节点
        bool is_first = false;
        #pragma omp critical
        {
            if (entry_node_ == -1) {
                entry_node_ = new_node_id;
                max_level_ = insert_level;
                is_first = true;
            }
        }
        if (is_first) return;

        int curr_entry_node;
        int curr_max_level;
        #pragma omp critical
        {
            curr_entry_node = entry_node_;
            curr_max_level = max_level_;
        }

        float d_entry = get_distance(
            data_ + (size_t)new_node_id * dim_,
            data_ + (size_t)curr_entry_node * dim_,
            dim_
        );
        MaxHeap enter_points;
        enter_points.push({d_entry, curr_entry_node});

        int curr_level = curr_max_level;
        VisitedList* vl = build_vl_pool_[tid]; 

        // 速降
        for (; curr_level > insert_level; --curr_level) {
            enter_points = searchLayer(data_ + (size_t)new_node_id * dim_, enter_points, 1, curr_level, vl);
        }

        // 逐层建立边
        for (; curr_level >= 0; --curr_level) {
            MaxHeap candidates = searchLayer(data_ + (size_t)new_node_id * dim_, enter_points, efConstruction_, curr_level, vl);   
            
            int max_m = (curr_level == 0) ? M0_ : M_;
            std::vector<int> chosen_neighbors = getNeighbors(candidates, max_m);
            
            // 构建新节点的出度
            int* my_links = getList(new_node_id, curr_level);
            for (size_t i = 0; i < chosen_neighbors.size(); ++i) {
                my_links[i + 1] = chosen_neighbors[i];
            }
            my_links[0] = chosen_neighbors.size();

            // 为目标邻居建立入度
            for (int neighbor_id : chosen_neighbors) {
                omp_set_lock(&node_locks_[neighbor_id]);

                int* neighbor_links = getList(neighbor_id, curr_level);
                int current_count = neighbor_links[0];
                
                if (current_count < max_m) {
                    neighbor_links[current_count + 1] = new_node_id;
                    neighbor_links[0]++;
                } else {
                    // 邻居的邻居数溢出了，进行启发式裁剪
                    MaxHeap neighbor_candidates;
                    neighbor_candidates.push({
                        get_distance(data_ + (size_t)neighbor_id * dim_, data_ + (size_t)new_node_id * dim_, dim_), 
                        new_node_id
                    });

                    for (int j = 1; j <= current_count; ++j) {
                        int nb = neighbor_links[j];
                        float d_nb = get_distance(data_ + (size_t)neighbor_id * dim_, data_ + (size_t)nb * dim_, dim_);
                        neighbor_candidates.push({d_nb, nb});
                    }  
                    
                    std::vector<int> trimmed_neighbors = getNeighbors(neighbor_candidates, max_m);
                    
                    // 先写数据块，最后刷新 Size
                    for (size_t j = 0; j < trimmed_neighbors.size(); ++j) {
                        neighbor_links[j + 1] = trimmed_neighbors[j];
                    }
                    neighbor_links[0] = trimmed_neighbors.size();
                }

                omp_unset_lock(&node_locks_[neighbor_id]);
            }
            enter_points = candidates;
        }

        // 如果该节点插入的层级超过了全图最高点，那么走全局临界区更新
        if (insert_level > curr_max_level) {
            #pragma omp critical
            {
                if (insert_level > max_level_) {
                    max_level_ = insert_level;
                    entry_node_ = new_node_id;
                }
            }
        }
    }

    /**
     * @brief KNN搜索主接口
     * @param vl 由调用线程传入实例指针
     */
    std::priority_queue<std::pair<float, uint32_t>> searchKnn(const float* query, size_t k, int efSearch, VisitedList* vl) {
        std::priority_queue<std::pair<float, uint32_t>> empty_res;
        
        if (entry_node_ == -1) {
            return empty_res;
        }

        float d_entry = get_distance(
            query,
            data_ + (size_t)entry_node_ * dim_,
            dim_
        );
        MaxHeap enter_points;
        enter_points.push({d_entry, entry_node_});

        for (int curr_level = max_level_; curr_level > 0; --curr_level) {
            enter_points = searchLayer(query, enter_points, 1, curr_level, vl);
        }

        MaxHeap search_res = searchLayer(query, enter_points, efSearch, 0, vl);

        while (search_res.size() > k) {
            search_res.pop();
        }

        std::priority_queue<std::pair<float, uint32_t>> final_k_res;
        while (!search_res.empty()) {
            auto item = search_res.top();
            search_res.pop();
            final_k_res.push({item.first, (uint32_t)item.second});
        }

        return final_k_res;
    }
};

} // namespace hnsw