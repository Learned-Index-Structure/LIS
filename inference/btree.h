#ifndef INFERENCE_BTREE_H
#define INFERENCE_BTREE_H

#include "btree/btree_map.h"
#include <vector>

using namespace std;
using tree_type = btree::btree_map<float, uint32_t>;

inline
uint32_t btree_find(const tree_type &tree, float key) {
    uint32_t pos = tree.find(key).position;
    return ((*(tree.find(key).node)).value(pos)).second;
}

inline
void btree_insert(tree_type &tree, const vector<float> &keys,
                  const vector<uint32_t> &values, uint32_t start_index, uint32_t end_index) {
    for (uint32_t i = start_index; i < end_index; i++) {
        tree.insert(std::pair<float, uint32_t>(keys[i], values[i]));
    }
}

#endif //INFERENCE_BTREE_H
