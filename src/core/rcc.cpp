#include "core/rcc.hpp"
#include "core/coreset.hpp"
#include <unordered_set>

namespace kmeans::core {

    RCC::RCC(int maxLevels) : max_levels(maxLevels) {
        levels.resize(std::max(1, max_levels));
    }

    void RCC::insertLeaf(const Coreset& leafCoreset) {
        auto carry = std::make_unique<RCCNode>(leafCoreset);

        if (levels.empty()) {
            levels.resize(std::max(1, this->max_levels));
        }

        for (int lvl = 0; lvl < static_cast<int>(levels.size()); ++lvl) {
            if (!levels[lvl]) {
                levels[lvl] = std::move(carry);
                break;
            } else {
                carry = mergeNodes(std::move(levels[lvl]), std::move(carry));
            }
        }

        if (carry) {
            if (levels.back()) {
                Coreset merged = mergeCoresets(levels.back()->coreset, carry->coreset); 
                carry = std::make_unique<RCCNode>(merged); 
            }
            levels.back() = std::move(carry);
        }
    }

    std::unique_ptr<RCCNode> RCC::mergeNodes(std::unique_ptr<RCCNode> nodeA, std::unique_ptr<RCCNode> nodeB) {
        if (!nodeA) return std::move(nodeB);
        if (!nodeB) return std::move(nodeA);

        Coreset merged = mergeCoresets(nodeA->coreset, nodeB->coreset); 
        auto parent = std::make_unique<RCCNode>(merged); 
        parent->left = std::move(nodeA);
        parent->right = std::move(nodeB);

        return parent;
    }

    Coreset RCC::getRootCoreset() const {
        Coreset final_coreset;
        bool empty = true;
        for (const auto& node : levels) {
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

    void RCC::clear() {
        levels.clear();
        levels.resize(std::max(1, max_levels));
    }

} // namespace kmeans::core