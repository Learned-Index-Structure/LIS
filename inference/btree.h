#ifndef INFERENCE_BTREE_H
#define INFERENCE_BTREE_H

#include "btree/btree_map.h"

btree::btree_map<int, int> tree;

inline int btree_find(int key) {
    int pos = tree.find(key).position;
    return ((*(tree.find(key).node)).value(pos)).second;
}

inline void btree_insert(int start_index,int end_index) {
    for (int i = start_index; i < end_index; i++) {
        tree.insert(std::pair<int, int>(i, i + 10));
    }
}


#endif //INFERENCE_BTREE_H
