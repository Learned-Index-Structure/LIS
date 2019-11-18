#ifndef INFERENCE_BTREE_HPP
#define INFERENCE_BTREE_HPP

#pragma once

#include "btree/btree_map.h"
#include <vector>

using namespace std;
using tree_type = btree::btree_map<uint64_t, uint32_t>;

inline
uint32_t btree_find(const tree_type &tree, uint64_t key) {
    auto it = tree.find(key);
    uint32_t pos = it.position;
    return ((*(it.node)).value(pos)).second;
}

inline
void btree_insert(tree_type &tree, const vector<uint64_t> &keys,
                  const vector<uint32_t> &values, uint32_t start_index, uint32_t end_index) {
    for (uint32_t i = start_index; i <= end_index; i++) {
        tree.insert(std::pair<uint64_t, uint32_t>(keys[i], values[i]));
    }
//    cout << "Btree size: " << tree.size() << endl;
}

#endif //INFERENCE_BTREE_HPP
