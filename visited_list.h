#pragma once
#include <vector>
#include <cstring>

namespace hnsw {

class VisitedList {
private:
    unsigned short* visited_array;   
    //使用unsigned short目的是已知明确当前的query数量为2000，可以根据实际修改
    //节约空间。同时，不使用bool的目的是，通过check当前visited_arrar[i]的值
    //是否等于当前查询批次，从而确定该节点是否被访问过。
    unsigned short current_mark;     
    //当前查询的query id的标号
    size_t num_elements;             

public:
    VisitedList(size_t num_elements) {
        this->num_elements = num_elements;
        visited_array = new unsigned short[num_elements];
        memset(visited_array, 0, num_elements * sizeof(unsigned short));  //提前开辟空间，避开频繁分配空间
        current_mark = 1;
    }

    ~VisitedList() {
        delete[] visited_array;
    }

    inline void advance() {
        current_mark++;
        if (current_mark == 0) {    //防止意外，正常情况current_mark初始化后不可能为0
            memset(visited_array, 0, num_elements * sizeof(unsigned short));
            current_mark = 1;
        }
    }

    inline void markVisited(size_t node_id) {   //标记这次被访问到的节点id
        visited_array[node_id] = current_mark;  
        //注意此处为直接将 current_mark赋值给visited_array[node_id]
        // 而不是++操作,使得始终可以保证一个节点在一个query内只被访问一次。
    }

    inline bool isVisited(size_t node_id) const {  //检查该节点是否被访问过
        return visited_array[node_id] == current_mark;
    }
};

} // namespace hnsw