#ifndef INFERENCE_BTREE_H
#define INFERENCE_BTREE_H

#include "btree/btree_map.h"
#include <vector>

using namespace std;
using tree_type = btree::btree_map<float, int>;

inline
int btree_find(const tree_type &tree, float key) {
    int pos = tree.find(key).position;
    return ((*(tree.find(key).node)).value(pos)).second;
}

inline
void btree_insert(tree_type tree, const vector<float> &keys,
                    const vector<int> &values, int start_index,int end_index) {
    for (int i = start_index; i < end_index; i++) {
        tree.insert(std::pair<float, int>(keys[i], values[i]));
    }
}

#endif //INFERENCE_BTREE_H
