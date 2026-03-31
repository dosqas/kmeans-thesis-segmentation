#include "core/rcc.hpp"
#include "core/coreset.hpp"
#include <unordered_set>

namespace kmeans {

    // Delete a subtree of our RCC. Used in cleaning up when deleting and pruning the RCC
    static void deleteSubtree(RCCNode* node)
    {
        if (!node) return;              // Base case - RCCNode is null, meaning we reached the end of the tree
        deleteSubtree(node->left);      // Recursively call the delete function on the left
        deleteSubtree(node->right);     // and right subtrees
        delete node;                    // Once we return from the recursion, we delete the node
    }

    // Insert a leaf in the RCC tree, merging as necessary to maintain structure
    void RCC::insertLeaf(const Coreset& leafCoreset, int sample_size)
    {
        RCCNode* carry = new RCCNode(leafCoreset); // New leaf node to insert

        if (levels.empty()) {
            levels.assign(std::max(1, this->max_levels), nullptr);
        }

        for (int lvl = 0; lvl < (int)levels.size(); ++lvl) {
            // If level is empty, place carry here and stop
            if (!levels[lvl]) {
                levels[lvl] = carry;
                carry = nullptr;
                break;
            }
            else {
                carry = mergeNodes(levels[lvl], carry); // Merge carry with existing node at this level
                levels[lvl] = nullptr;
            }
        }

        // If still have a carry beyond max_levels, drop oldest by merging into last level
        if (carry) {
            // bounded cap: merge into top level and replace
            if (levels.back()) {
                RCCNode* old_top = levels.back();
                Coreset merged = mergeCoresets(old_top->coreset, carry->coreset); // Merge with existing top level
                deleteSubtree(old_top); // Delete the old top level subtree
                deleteSubtree(carry);   // Delete the carry subtree
                carry = new RCCNode(merged); // Create a flattened replacement
            }
            levels.back() = carry;
        }
    }

    // Merge two RCC nodes A and B into one by merging their coresets and creating a new parent node
    // having A and B as children
    RCCNode* RCC::mergeNodes(RCCNode* nodeA, RCCNode* nodeB)
    {
        if (!nodeA) return nodeB;
        if (!nodeB) return nodeA;

        Coreset merged = mergeCoresets(nodeA->coreset, nodeB->coreset); // Merge the coresets of A and B
        RCCNode* parent = new RCCNode(merged); // Create a new parent node with the merged coreset
        parent->left = nodeA;
        parent->right = nodeB;

        return parent;
    }

    // Get the coreset at the root of the RCC tree
    Coreset RCC::getRootCoreset() const
    {
        Coreset final_coreset;
        bool empty = true;
        for (auto node : levels) {
            if (node) {
                if (empty) {
                    final_coreset = node->coreset;
                    empty = false;
                } else {
                    final_coreset = mergeCoresets(final_coreset, node->coreset);
                }
            }
        }
        return final_coreset;
    }

    void RCC::clear()
    {
        for (RCCNode* node : levels) {
            if (node) {
                deleteSubtree(node);
            }
        }
        levels.assign(std::max(1, max_levels), nullptr);
    }

    RCC::~RCC()
    {
        clear();
    }
}