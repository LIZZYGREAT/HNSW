#pragma once
#include <vector>
#include <queue>
#include <random>
#include <algorithm.h>
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
    float* data_;      // 底层向量数据的指针

    int max_level_;         // 当前整个图的最高层数
    int entry_node_;   // 最高层的入口节点 ID

    // 记录每个节点的最大层数
    int* node_max_level_; 
    
    int*** links_;   //核心参数 
    // 邻接表：links_[node_id][level] 指向一个动态数组，结构为：[node_id][level][neighbor_id]
    //其中，node_id为节点id，level为层数，neighbor_id为邻居节点id。
    // 数组的第 0 个元素存 "当前邻居数量"，后续元素存 "邻居的 ID"
    // 第 0 层数组长度为 M0_ + 1，其他层为 M_ + 1 

    // 随机数生成器决定节点层数
    std::default_random_engine random_generator_;
    std::uniform_real_distribution<double> uniform_distribution_;

    // 记录节点是否被访问过
    VisitedList* visited_list_;

    // 每次 pop 出最小距离。
    typedef std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> MinHeap;

    //每次 pop 出最大距离。
    typedef std::priority_queue<std::pair<float, int>> MaxHeap;

public:
    HNSW(float* data, size_t node_num, size_t dim, int M, int efConstruction) 
        : data_(data), node_num_(node_num), dim_(dim), 
          M_(M), efConstruction_(efConstruction) {
        
        M0_ = M_ * 2;
        mult_ = 1.0 / log(1.0 * M_);
        max_level_ = -1; 
        entry_node_ = -1;

        // 初始化随机数生成器
        uniform_distribution_ = std::uniform_real_distribution<double>(0.0, 1.0);

        // 分配节点层数记录数组
        node_max_level_ = new int[node_num_];

        // 分配并初始化邻接表
        links_ = new int**[node_num_];
        for (size_t i = 0; i < node_num_; ++i) {
            links_[i] = nullptr; 
        }

        visited_list_ = new VisitedList(node_num_);
    }

    ~HNSW() {
        // 释放内存
        delete visited_list_;
        delete[] node_max_level_;
        delete[] links_;
    }

    // 生成随机层数
    int getRandomLevel() {
        double r = uniform_distribution_(random_generator_);
        if (r < std::numeric_limits<double>::epsilon()) {   
            r = 1.0; 
        }
        return (int)(-log(r) * mult_);  //计算随机层数
    }

private:
    /** 
     * @brief 在指定的 level 层内进行贪心搜索
     * @param query 查询向量
     * @param enter_points 当前层的入口节点集合 (大顶堆格式，因为是从上一层的结果传下来的)
     * @param ef 需要搜索的候选数量 (即结果队列的最大容量)
     * @param level 当前所在的层级
     * @return MaxHeap 包含当前层找到的离 query 最近的最多 ef 个节点
     */
    MaxHeap searchLayer(const float* query, MaxHeap enter_points, int ef, int level) {
        VisitedList* vl = visited_list_;
        vl->advance();

        MinHeap candidates; // 候选小顶堆 (C)
        MaxHeap results;    // 结果大顶堆 (W)

        // 将入口节点加入 candidates 和 results，并标记为已访问
        while (!enter_points.empty()) {
            std::pair<float, int> ep = enter_points.top();
            enter_points.pop();
            
            candidates.push(ep);
            results.push(ep);
            vl->markVisited(ep.second);
        }

        // 贪心搜索
        while (!candidates.empty()) {
            // 从候选池中拿出当前离 query 最近的节点 c
            std::pair<float, int> curr_c = candidates.top();
            candidates.pop();

            float dist_c = curr_c.first;
            int node_c = curr_c.second;

            // 提取结果集中最远的距离
            float dist_f = results.top().first;

            // 如果连候选池里最近的那个点，都比结果集里最远的那个点还要远，则跳过本次搜索
            if (dist_c > dist_f) {
                break; 
            }

            // 遍历节点 c 在当前层的邻居
            int* neighbors = links_[node_c][level];
            if (neighbors == nullptr) {
                continue; // 这个节点在这一层没有邻居
            }

            int num_neighbors = neighbors[0]; // 数组第0位存的是邻居数量

            for (int i = 1; i <= num_neighbors; ++i) {
                int neighbor_id = neighbors[i];

                // 如果邻居还没被访问过
                if (!vl->isVisited(neighbor_id)) {
                    vl->markVisited(neighbor_id);

                    // 计算邻居与 query 的距离
                    float dist_e = get_distance(
                        query, 
                        data_ + (size_t)neighbor_id * dim_, 
                        dim_
                    );

                    // 如果结果集还没满，或者新邻居比结果集里最远的那个更近
                    if (results.size() < (size_t)ef || dist_e < results.top().first) {
                        candidates.push({dist_e, neighbor_id});
                        results.push({dist_e, neighbor_id});

                        if (results.size() > (size_t)ef) {
                            results.pop();   //如果满了，就踢出距离最远的
                        }
                    }
                }
            }
        }

        return results;
    }

    // 启发式选边，距离远近以及方向，尽量避开方向一致且距离较近的重复信息的节点。

    // 这里有个小细节在于，我们的启发式选边，看似没有将query/insert_node_id作为参数传入
    // 但是，在candidates中其实就已经包含了候选节点到query/insert_node_id的距离信息
    // 只需要我们在调用该函数前实现得到已处理好的candidates即可。从而实现了逻辑层面的解耦合
    std::vector<int> getNeighbors(MaxHeap candidates, int M_max) {
        if (candidates.size() <= (size_t)M_max) {    
            //候选节点数不足最大邻居数
            std::vector<int> res;
            std::vector<std::pair<float, int>> temp;

            while (!candidates.empty()) {
                temp.push_back(candidates.top());
                candidates.pop();
            }
            std::reverse(temp.begin(), temp.end()); // 倒序成距离从小到大
            for (auto& p : temp) res.push_back(p.second);
            return res;
        }

        // 把大顶堆转化为距离从小到大的有序向量
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
            float dist_to_candidate = c.first;  //insert_node_id到候选节点的距离

            bool keep = true;
            // 新候选节点是否离已选邻居里的某个节点更近
            for (int result_neighbor_id : result_neighbors) {
                float dist_to_neighbor = get_distance(
                    data_ + (size_t)candidate_id * dim_,
                    data_ + (size_t)result_neighbor_id * dim_,
                    dim_
                );
                // 如果新候选节点离某个已选邻居更近，说明那个方向被覆盖了，信息冗余
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
    // addPoint在做的事情是：将新节点插入到图中，并更新图的结构。其实就是建立邻接链表Links_
    // 为什么需要速降？高层稀疏，所以能快速进行大致定位，就好像图书馆里面找书，先确认在哪个分区，哪个书架，第几层
    // 而且直接找新插入结点的最近的节点，会导致图为容易出现孤立，需要我们进行启发找边。
    // 速降的本质是利用searchLayer函数，将需要插入的节点作为query，ef取1实现快速下降到距离插入节点附近的节点。
    // 然后再通过候选节点，通过上述启发式的getNeighbors函数，得到适合的插入节点的邻居节点。
    // 注意，我们不仅需要对insert_level进行建立nerighbors,实际上需要我们对从第0层到insert_level层都进行链接。

    void addPoint(int new_node_id) {
        // 随机数决定当前节点的最大层数
        int insert_level = getRandomLevel();
        node_max_level_[new_node_id] = insert_level;  //记录当前节点的最大层数

        links_[new_node_id] = new int*[insert_level + 1];
        for (int l = 0; l <= insert_level; ++l) {
            int max_m = (l == 0) ? M0_ : M_;
            links_[new_node_id][l] = new int[max_m + 1];
            links_[new_node_id][l][0] = 0; // 初始邻居数为0
        }

        // 如果是全场第一个点，直接为入口
        if (entry_node_ == -1) {
            entry_node_ = new_node_id;
            max_level_ = insert_level;
            return;
        }

        // 构建速降的初始入口节点
        float d_entry = get_distance(
            data_ + (size_t)new_node_id * dim_,
            data_ + (size_t)entry_node_ * dim_,
            dim_
        );
        MaxHeap enter_points;
        enter_points.push({d_entry, entry_node_});

        // 速降，从最高层到新节点层的上一层，每层只找1个最近点，实现快速下降
        int curr_level = max_level_;
        for (; curr_level > insert_level; --curr_level) {
            enter_points = searchLayer(data_ + (size_t)new_node_id * dim_, enter_points, 1, curr_level);
        }

        // 链接边，从insert_level层到0层，逐层建立边
        for (; curr_level >= 0; --curr_level) {
            MaxHeap candidates = searchLayer(data_ + (size_t)new_node_id * dim_, 
            enter_points, efConstruction_, curr_level);   
            
            int max_m = (curr_level == 0) ? M0_ : M_;
            std::vector<int> chosen_neighbors = getNeighbors(candidates, max_m);//启发式找边
            
            links_[new_node_id][curr_level][0] = chosen_neighbors.size();
            for (size_t i = 0; i < chosen_neighbors.size(); ++i) {
                links_[new_node_id][curr_level][i + 1] = chosen_neighbors[i];
            }

            // 反向连线与溢出裁剪
            for (int neighbor_id : chosen_neighbors) {
                int* neighbor_links = links_[neighbor_id][curr_level];
                int current_count = neighbor_links[0];
                
                neighbor_links[current_count + 1] = new_node_id;
                neighbor_links[0]++;

                // 如果邻居的邻居数溢出了，进行裁剪
                if (neighbor_links[0] > max_m) {
                    MaxHeap neighbor_candidates;

                    for (int j = 1; j <= neighbor_links[0]; ++j) {
                        int nb = neighbor_links[j];
                        float d_nb = get_distance(
                            data_ + (size_t)neighbor_id * dim_,
                            data_ + (size_t)nb * dim_,
                            dim_
                        );
                        neighbor_candidates.push({d_nb, nb});
                    }  //处理得到getNeighbors函数需要的最大堆candidates
                    std::vector<int> trimmed_neighbors = getNeighbors(neighbor_candidates, max_m);
                    neighbor_links[0] = trimmed_neighbors.size();
                    for (size_t j = 0; j < trimmed_neighbors.size(); ++j) {
                        neighbor_links[j + 1] = trimmed_neighbors[j];
                    }
                }
            }
            // 将当前层的候选池无缝传给下一层作为入口种子
            enter_points = candidates;
        }

        // 如果插入层大于当前最大层数，更新最大层数和入口节点
        if (insert_level > max_level_) {
            max_level_ = insert_level;
            entry_node_ = new_node_id;
        }
    }

public:
    /**
     * @brief KNN搜索主接口
     * @param query 查询向量的指针
     * @param k 需要返回的最近邻个数
     * @param efSearch 查询时的控制池大小 (efSearch >= k)，该值越大准确率越高，但耗时也越大
     * @return std::priority_queue<std::pair<float, uint32_t>> 返回大顶堆
     */
    std::priority_queue<std::pair<float, uint32_t>> searchKnn(const float* query, size_t k, int efSearch) {
        std::priority_queue<std::pair<float, uint32_t>> empty_res;
        
        if (entry_node_ == -1) {
            return empty_res;
        }

        // 全局起点
        float d_entry = get_distance(
            query,
            data_ + (size_t)entry_node_ * dim_,
            dim_
        );
        MaxHeap enter_points;
        enter_points.push({d_entry, entry_node_});

        // 速降,从最高层一直垂直下降到第 1 层，每层 ef=1，快速接近目标区域
        for (int curr_level = max_level_; curr_level > 0; --curr_level) {
            enter_points = searchLayer(query, enter_points, 1, curr_level);
        }

        // 第 0 层精细搜索
        MaxHeap search_res = searchLayer(query, enter_points, efSearch, 0);

        // .top() 是当前池子里距离最远的那个，我们只需不断弹出最差的，直到池子大小刚好缩到 k为止
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