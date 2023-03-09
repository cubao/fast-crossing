#ifndef CUBAO_KD_QUIVER_HPP
#define CUBAO_KD_QUIVER_HPP

#include "nanoflann_kdtree.hpp"
#include "quiver.hpp"
#include "polyline_ruler.hpp"

namespace cubao {
inline std::pair<int, double> range_to_seg_t(const Eigen::VectorX &ranges, double range)
{
    // TODO, merge into polyline_ruler
    // use lower_bound?
    int seg_idx = 0;
    double t = 0.0;
    int N_ = ruler.size();
    if (range <= 0.0) {
        t = range / ranges[1];
    } else if (range >= ranges[N_-1]) {
        t = (range - ranges[N_ - 2]) / (ranges[N_ - 1] - ranges[N_ - 2]);
    } else {
        while (seg_idx + 1 < N_ && ranges[seg_idx + 1] < range) {
            ++seg_idx;
        }
        t = (range - ranges[seg_idx]) / (ranges[seg_idx + 1] - ranges[seg_idx]);
    }
    return {seg_idx, t};
}

struct KdQuiver : Quiver {

    // add data
    void add(const std::map<int, PolylineRuler> &rulers) {
        for (auto &pair: rulers) {
            add(pair.second, pair.first);
        }
    }
    void add(const PolylineRuler &ruler, int polyline_index) {
        auto &xyzs = ruler.polyline();
        if (is_wgs84_) {
            tree.add(lla2enu(polyline, anchor_));
        } else {
            tree.add(xyzs);
        }
        for (int i = 0; i < N; ++i) {
            index(polyline_index, i);
        }
    }

    // helpers
    Arrow arrow(const PolylineRuler &ruler, int segment_index, double t) {
        auto [pos, dir]  = ruler.arrow(range);
        auto &polyline_ = ruler.polyline();
        Eigen::Vector3d pos = ruler.interpolate(polyline_.row(segment_index), polyline_.row(segment_index + 1), t, ruler.is_wgs84());
        Eigen::Vector3d dir = ruler.dirs().row(segment_index);
        return Arrow(pos, dir).segment_index(segment_index).t(t);
    }
    Arrow arrow(const PolylineRuler &ruler, double range) {
        auto [seg_idx, t] = range_to_seg_t(ruler.ranges(), range);
        return arrow(ruler, seg_idx, t);
    }

    void reset() {
        tree_.reset();
        index_list_.clear();
        index_map_.clear();
    }

private:
    KdTree tree_;
    int index(int polyline_index, int segment_index) {
        auto [itr, new_insert] = index_map_[polyline_index].emplace(segment_index, index_list_.size());
        if (new_insert) {
            index_list_.emplace_back(polyline_index, segment_index);
        }
        return itr->second;
    }
    Eigen::Vector2i index(int index) {
        return index_list_[index];
    }
    std::vector<Eigen::Vector2i> index_list_; // polyline_index, segment_index
    std::map<int, std::map<int, int>> index_map_;
};
}

#endif