#ifndef CUBAO_KD_QUIVER_HPP
#define CUBAO_KD_QUIVER_HPP

#include "nanoflann_kdtree.hpp"
#include "quiver.hpp"
#include "polyline_ruler.hpp"

namespace cubao
{
inline std::pair<int, double> range_to_seg_t(const Eigen::VectorXd &ranges,
                                             double range)
{
    // TODO, merge into polyline_ruler
    // use lower_bound?
    int seg_idx = 0;
    double t = 0.0;
    int N_ = ranges.size();
    if (range <= 0.0) {
        t = range / ranges[1];
    } else if (range >= ranges[N_ - 1]) {
        t = (range - ranges[N_ - 2]) / (ranges[N_ - 1] - ranges[N_ - 2]);
    } else {
        while (seg_idx + 1 < N_ && ranges[seg_idx + 1] < range) {
            ++seg_idx;
        }
        t = (range - ranges[seg_idx]) / (ranges[seg_idx + 1] - ranges[seg_idx]);
    }
    return {seg_idx, t};
}

inline bool is_in_range(double v, const Eigen::VectorXd &intervals)
{
    for (int i = 0, N = intervals.size(); i + 1 < N; i += 2) {
        if (intervals[i] <= v && v <= intervals[i + 1]) {
            return true;
        }
    }
    return false;
}

struct KdQuiver : Quiver
{
    KdQuiver() : Quiver() {}
    KdQuiver(const Eigen::Vector3d &anchor_lla) : Quiver(anchor_lla) {}

    int add(const RowVectors &polyline, int index = -1)
    {
        if (index < 0) {
            index = polylines_.size();
        }
        if (polylines_.find(index) != polylines_.end()) {
            throw std::invalid_argument("duplicate index: " +
                                        std::to_string(index));
        }
        polylines_.emplace(index, PolylineRuler(polyline, is_wgs84_));
        tree_.reset();
        return index;
    }
    int add(const Eigen::Ref<const RowVectorsNx2> &polyline, int index = -1)
    {
        RowVectors Nx3(polyline.rows(), 3);
        Nx3.leftCols<2>() = polyline;
        Nx3.col(2).setZero();
        return add(Nx3, index);
    }

    void build(bool force = false) const
    {
        if (!force && tree_) {
            return;
        }
        reset_index();
        // auto &xyzs = ruler.polyline();
        // const int N = ruler.N();
        // if (is_wgs84_) {
        //     tree_.add(lla2enu(xyzs));
        // } else {
        //     tree_.add(xyzs);
        // }
        // for (int i = 0; i < N; ++i) {
        //     index(polyline_index, i);
        // }
    }

    // query
    std::pair<int, double> nearest(const Eigen::Vector3d &position, //
                                   bool return_squared_l2 = false) const
    {
        return tree().nearest(is_wgs84_ ? lla2enu(position) : position,
                              return_squared_l2);
    }
    std::pair<int, double> nearest(int index, //
                                   bool return_squared_l2 = false) const
    {
        return tree().nearest(index, return_squared_l2);
    }
    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    nearest(const Eigen::Vector3d &position, //
            int k,                           //
            bool sorted = true,              //
            bool return_squared_l2 = false) const
    {
        return tree().nearest(is_wgs84_ ? lla2enu(position) : position, //
                              k,                                        //
                              sorted,                                   //
                              return_squared_l2);
    }
    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    nearest(const Eigen::Vector3d &position, //
            double radius,                   //
            bool sorted = true,              //
            bool return_squared_l2 = false) const
    {
        return tree().nearest(is_wgs84_ ? lla2enu(position) : position, //
                              radius,                                   //
                              sorted,                                   //
                              return_squared_l2);
    }

    RowVectors positions(const Eigen::VectorXi &hits) const
    {
        const int N = hits.size();
        RowVectors coords(N, 3);
        for (int i = 0; i < N; ++i) {
            // index(hits[i]);
        }
        return coords;
    }
    RowVectors directions(const Eigen::VectorXi &hits) const
    {
        const int N = hits.size();
        RowVectors coords(N, 3);
        return coords;
    }

    // Eigen::VectorXi
    // filter(const Eigen::VectorXi &hits, const Arrow &arrow,
    //        // positions
    //        std::optional<Eigen::VectorXd> x_slots = std::nullopt,
    //        std::optional<Eigen::VectorXd> y_slots = std::nullopt,
    //        std::optional<Eigen::VectorXd> z_slots = std::nullopt, )
    // {
    // }

    // helpers
    Arrow arrow(const PolylineRuler &ruler, int segment_index, double t)
    {

        auto &polyline_ = ruler.polyline();
        Eigen::Vector3d pos =
            ruler.interpolate(polyline_.row(segment_index),     //
                              polyline_.row(segment_index + 1), //
                              t, ruler.is_wgs84());
        Eigen::Vector3d dir = ruler.dirs().row(segment_index);
        auto arrow = Arrow(pos, dir);
        arrow.segment_index(segment_index).t(t);
        if (is_wgs84_ && ruler.is_wgs84()) {
            arrow.position(lla2enu(arrow.position()));
        }
        return arrow;
    }
    Arrow arrow(const PolylineRuler &ruler, double range)
    {
        auto [seg_idx, t] = range_to_seg_t(ruler.ranges(), range);
        return arrow(ruler, seg_idx, t);
    }

    void reset() { reset_index(); }

    void reset_index() const
    {
        tree_.reset();
        index_list_.clear();
        index_map_.clear();
    }

    int index(int polyline_index, int segment_index)
    {
        auto [itr, new_insert] = index_map_[polyline_index].emplace(
            segment_index, index_list_.size());
        if (new_insert) {
            index_list_.emplace_back(polyline_index, segment_index);
        }
        return itr->second;
    }
    Eigen::Vector2i index(int index) { return index_list_[index]; }

    const std::map<int, PolylineRuler> &polylines() const { return polylines_; }
    const PolylineRuler *polyline(int index) const
    {
        auto itr = polylines_.find(index);
        if (itr == polylines_.end()) {
            return nullptr;
        }
        return &itr->second;
    }

  private:
    // data
    std::map<int, PolylineRuler> polylines_;

    // index
    mutable std::optional<KdTree> tree_;
    KdTree &tree() const
    {
        build();
        return *tree_;
    }
    // polyline_index, segment_index
    mutable std::vector<Eigen::Vector2i> index_list_;
    mutable std::map<int, std::map<int, int>> index_map_;
};
} // namespace cubao

#endif
