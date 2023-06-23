// should sync
// - https://github.com/cubao/fast-crossing/blob/master/src/kd_quiver.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/kd_quiver.hpp

#ifndef CUBAO_KD_QUIVER_HPP
#define CUBAO_KD_QUIVER_HPP

#include "nanoflann_kdtree.hpp"
#include "quiver.hpp"
#include "polyline_ruler.hpp"

namespace cubao
{
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
        return add(to_Nx3(polyline), index);
    }

    void build(bool force = false) const
    {
        if (!force && tree_) {
            return;
        }
        reset_index();
        tree_ = KdTree();
        for (auto &pair : polylines_) {
            int polyline_idx = pair.first;
            auto &xyzs = pair.second.polyline();
            tree_->add(is_wgs84_ ? lla2enu(xyzs) : xyzs);
            for (int pt_idx = 0, N = xyzs.rows(); pt_idx < N; ++pt_idx) {
                __index(polyline_idx, pt_idx);
            }
        }
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
            bool sort = true,                //
            bool return_squared_l2 = false) const
    {
        return tree().nearest(is_wgs84_ ? lla2enu(position) : position, //
                              k,                                        //
                              sort,                                     //
                              return_squared_l2);
    }
    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    nearest(const Eigen::Vector3d &position, //
            double radius,                   //
            bool sort = true,                //
            bool return_squared_l2 = false) const
    {
        return tree().nearest(is_wgs84_ ? lla2enu(position) : position, //
                              radius,                                   //
                              sort,                                     //
                              return_squared_l2);
    }

    Eigen::Map<const RowVectors> positions() const { return tree().points(); }

    RowVectors positions(const Eigen::VectorXi &hits) const
    {
        const int N = hits.size();
        RowVectors coords(N, 3);
        for (int i = 0; i < N; ++i) {
            auto _ = index(hits[i]);
            int line_idx = _[0];
            int pt_idx = _[1];
            auto &ruler = polylines_.at(line_idx);
            coords.row(i) = ruler.at(i);
        }
        return coords;
    }
    RowVectors directions(const Eigen::VectorXi &hits) const
    {
        const int N = hits.size();
        RowVectors coords(N, 3);
        for (int i = 0; i < N; ++i) {
            auto _ = index(hits[i]);
            int line_idx = _[0];
            int pt_idx = _[1];
            auto &ruler = polylines_.at(line_idx);
            coords.row(i) = ruler.dir(i);
        }
        return coords;
    }
    std::vector<Arrow> arrows(const Eigen::VectorXi &hits) const
    {
        const int N = hits.size();
        std::vector<Arrow> arrows;
        arrows.reserve(N);
        for (int i = 0; i < N; ++i) {
            auto _ = index(hits[i]);
            int line_idx = _[0];
            int pt_idx = _[1];
            arrows.push_back(arrow(line_idx, pt_idx));
        }
        return arrows;
    }

    Arrow arrow(const PolylineRuler &ruler, int segment_index, double t) const
    {

        auto &polyline_ = ruler.polyline();
        Eigen::Vector3d pos =
            ruler.interpolate(polyline_.row(segment_index),     //
                              polyline_.row(segment_index + 1), //
                              t);
        Eigen::Vector3d dir = ruler.dirs().row(segment_index);
        auto arrow = Arrow(pos, dir);
        arrow.segment_index(segment_index).t(t);
        if (is_wgs84_ && ruler.is_wgs84()) {
            arrow.position(lla2enu(arrow.position()));
        }
        return arrow;
    }
    Arrow arrow(const PolylineRuler &ruler, double range) const
    {
        auto [seg_idx, t] = ruler.segment_index_t(range);
        return arrow(ruler, seg_idx, t);
    }
    Arrow arrow(int point_index) const
    {
        Eigen::Vector2i ii = this->index(point_index);
        return arrow(ii[0], ii[1]);
    }
    Arrow arrow(int polyline_index, int segment_index) const
    {
        auto ruler = polyline(polyline_index);
        return Arrow(ruler->at(segment_index), ruler->dir(segment_index)) //
            .polyline_index(polyline_index)                               //
            .segment_index(segment_index)                                 //
            .t(0.0)                                                       //
            .range(ruler->range(segment_index))                           //
            ;
    }
    Arrow arrow(int polyline_index, int segment_index, double t) const
    {
        auto ruler = polyline(polyline_index);
        return Arrow(ruler->at(segment_index, t), ruler->dir(segment_index))
            .polyline_index(polyline_index)        //
            .segment_index(segment_index)          //
            .t(t)                                  //
            .range(ruler->range(segment_index, t)) //
            ;
    }
    Arrow arrow(int polyline_index, double range) const
    {
        auto ruler = polyline(polyline_index);
        auto [xyz, dir] = ruler->arrow(range, false);
        auto [idx, t] = ruler->segment_index_t(range);
        return Arrow(xyz, dir)              //
            .polyline_index(polyline_index) //
            .segment_index(idx)             //
            .t(t)                           //
            .range(range)                   //
            ;
    }

    static Eigen::VectorXi filter(const std::vector<Arrow> &arrows,
                                  const Arrow &arrow,
                                  const Quiver::FilterParams &params,
                                  bool is_wgs84 = false)
    {
        if (is_wgs84) {
            Quiver quiver(arrow.position());
            return quiver.filter(arrows, arrow, params);
        } else {
            Quiver quiver;
            return quiver.filter(arrows, arrow, params);
        }
    }

    static Eigen::VectorXi select_by_mask(const Eigen::VectorXi &indexes,
                                          const Eigen::VectorXi &mask)
    {
        std::vector<int> ret;
        ret.reserve(mask.sum());
        for (int i = 0, N = indexes.size(); i < N; ++i) {
            if (mask[i]) {
                ret.push_back(indexes[i]);
            }
        }
        return Eigen::VectorXi::Map(&ret[0], ret.size());
    }
    static Eigen::VectorXd select_by_mask(const Eigen::VectorXd &norms,
                                          const Eigen::VectorXi &mask)
    {
        std::vector<double> ret;
        ret.reserve(mask.sum());
        for (int i = 0, N = norms.size(); i < N; ++i) {
            if (mask[i]) {
                ret.push_back(norms[i]);
            }
        }
        return Eigen::VectorXd::Map(&ret[0], ret.size());
    }

    Eigen::VectorXi filter(const Eigen::VectorXi &hits, //
                           const Arrow &arrow,          //
                           const Quiver::FilterParams &params) const
    {
        auto mask = filter(arrows(hits), arrow, //
                           params, is_wgs84_);
        return select_by_mask(hits, mask);
    }

    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    filter(const Eigen::VectorXi &hits,  //
           const Eigen::VectorXd &norms, //
           const Arrow &arrow,           //
           const Quiver::FilterParams &params) const
    {
        auto mask = filter(arrows(hits), arrow, //
                           params, is_wgs84_);
        return {select_by_mask(hits, mask), //
                select_by_mask(norms, mask)};
    }

    void reset() { reset_index(); }

    void reset_index() const
    {
        tree_.reset();
        index_list_.clear();
        index_map_.clear();
    }

    int index(int polyline_index, int segment_index) const
    {
        return index_map_.at(polyline_index).at(segment_index);
    }
    Eigen::Vector2i index(int index) const { return index_list_[index]; }

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
    int __index(int polyline_index, int segment_index) const
    {
        auto [itr, new_insert] = index_map_[polyline_index].emplace(
            segment_index, index_list_.size());
        if (new_insert) {
            index_list_.emplace_back(polyline_index, segment_index);
        }
        return itr->second;
    }
};
} // namespace cubao

#endif
