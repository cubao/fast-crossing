// should sync
// - https://github.com/cubao/fast-crossing/blob/master/src/fast_crossing.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/fast_crossing.hpp

#ifndef CUBAO_FAST_CROSSING_HPP
#define CUBAO_FAST_CROSSING_HPP

#include <set>
#include <vector>
#include <optional>
#include <limits>
#include <cmath>

#include "flatbush.h"
#include "polyline_ruler.hpp"
#include "kd_quiver.hpp"
#include "point_in_polygon.hpp"

namespace cubao
{
constexpr double PI = 3.14159265358979323846;
using IntNx2 = Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>;

// https://github.com/isl-org/Open3D/blob/88693971ae7a7c3df27546ff7c5b1d91188e39cf/cpp/open3d/utility/Helper.h#L71
template <typename T> struct hash_eigen
{
    std::size_t operator()(T const &matrix) const
    {
        size_t hash_seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            hash_seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                         (hash_seed << 6) + (hash_seed >> 2);
        }
        return hash_seed;
    }
};

inline void sort_indexes(std::vector<Eigen::Vector2i> &indexes)
{
    std::sort(indexes.begin(), indexes.end(),
              [](const auto &idx1, const auto &idx2) {
                  if (idx1[0] == idx2[0]) {
                      return idx1[1] < idx2[1];
                  }
                  return idx1[0] < idx2[0];
              });
}

// Parameters of the perpendicular scanline ("横向候选") query.
//
// Semantics, matching the legacy
// `tbc_bentley_ottmann::ScanlineHitFilterOptions` so that a batch result can be
// compared element-wise against a hand-written per-point loop:
//
//   max_height_offset > 0   drop hits with |lla.z - query.z| beyond it.
//                           0 (or negative) disables the check.
//   min_dot, accept_opposite are jointly defined:
//     accept_opposite=false: keep hits with dot >= min_dot, i.e. "same
//                            direction only". A negative min_dot disables it.
//     accept_opposite=true : keep hits with dot >= min_dot OR
//                            dot <= -min_dot, i.e. "same or opposite".
//                            Here min_dot is a gate on |dot|: min_dot=0.9
//                            keeps both senses within ~+-25.8 deg of the
//                            query direction and rejects perpendicular hits.
//                            min_dot <= 0 disables the check.
//   filter_out_hits_by_gap > 0
//                           keep only the cluster of hits reachable from
//                           lateral offset 0 without a gap larger than this
//                           (nearest lane group, drops the "other road").
//                           Hits are ordered by |lateral| before this filter,
//                           so the cluster is taken from a sorted sequence.
//   only_nearest            keep only the hit with the smallest |lateral|.
//
// Filters run in this order: height -> direction -> gap -> only_nearest.
struct ScanlineHitFilterOptions
{
    double max_height_offset{3.0};
    double min_dot{0.5};
    bool accept_opposite{false};
    double filter_out_hits_by_gap{0.0};
    bool only_nearest{false};
};

// One segment hit by a scanline.
struct ScanlineHit
{
    int polyline_index{-1};
    int segment_index{-1};
    double t{0};                        // parameter inside the segment
    Eigen::Vector3d lla{0.0, 0.0, 0.0}; // hit point
    Eigen::Vector3d dir{0.0, 0.0, 1.0}; // segment direction (unit, z = 0)
    double lateral_offset{0.0};         // signed, left positive
    double vertical_offset{0.0};
    double dot{0.0}; // dot(query dir, segment dir)
};

// |lateral| ascending, ties broken by (polyline_index, segment_index) so that
// the order is deterministic (std::sort is not stable).
inline void sort_by_abs_lateral_offset(std::vector<ScanlineHit> &hits)
{
    std::sort(hits.begin(), hits.end(),
              [](const ScanlineHit &h1, const ScanlineHit &h2) {
                  double a1 = std::fabs(h1.lateral_offset);
                  double a2 = std::fabs(h2.lateral_offset);
                  if (a1 != a2) {
                      return a1 < a2;
                  }
                  if (h1.polyline_index != h2.polyline_index) {
                      return h1.polyline_index < h2.polyline_index;
                  }
                  return h1.segment_index < h2.segment_index;
              });
}

// [i, j] of `values` (ascending) reachable from 0 with gaps <= max_gap.
inline std::pair<int, int> reachable_range(const std::vector<double> &values,
                                           double max_gap)
{
    if (values.empty()) {
        return {0, -1};
    }
    const int N = static_cast<int>(values.size());
    if (values[0] >= 0) {
        if (values[0] > max_gap) {
            return {0, -1};
        }
        int j = 0;
        while (j + 1 < N && values[j + 1] - values[j] <= max_gap) {
            ++j;
        }
        return {0, j};
    }
    if (values[N - 1] <= 0) {
        if (values[N - 1] < -max_gap) {
            return {0, -1};
        }
        int i = N - 1;
        while (i - 1 >= 0 && values[i] - values[i - 1] <= max_gap) {
            --i;
        }
        return {i, N - 1};
    }
    int j = 0;
    while (j < N && values[j] < 0) {
        ++j;
    }
    int i = j - 1;
    if (j < N && values[j] > max_gap) {
        --j;
    } else {
        while (j + 1 < N && values[j + 1] - values[j] <= max_gap) {
            ++j;
        }
    }
    if (i >= 0 && values[i] < -max_gap) {
        ++i;
    } else {
        while (i - 1 >= 0 && values[i] - values[i - 1] <= max_gap) {
            --i;
        }
    }
    return {i, j};
}

// Keep the contiguous group of hits around lateral offset 0 whose neighbouring
// gaps are all <= max_gap (i.e. the nearest lane group; drops "the road over
// there"). `reachable_range` needs *signed* lateral offsets in ascending order,
// so this sorts by signed offset rather than by |offset|.
inline void filter_scanline_hits_by_gap(std::vector<ScanlineHit> &hits,
                                        double max_gap)
{
    if (hits.empty()) {
        return;
    }
    std::vector<ScanlineHit> sorted = hits;
    std::sort(sorted.begin(), sorted.end(),
              [](const ScanlineHit &h1, const ScanlineHit &h2) {
                  if (h1.lateral_offset != h2.lateral_offset) {
                      return h1.lateral_offset < h2.lateral_offset;
                  }
                  if (h1.polyline_index != h2.polyline_index) {
                      return h1.polyline_index < h2.polyline_index;
                  }
                  return h1.segment_index < h2.segment_index;
              });
    std::vector<double> offsets(sorted.size());
    for (int i = 0, N = static_cast<int>(sorted.size()); i < N; ++i) {
        offsets[i] = sorted[i].lateral_offset;
    }
    auto [i, j] = reachable_range(offsets, max_gap);
    if (i > j) {
        hits.clear();
        return;
    }
    hits = std::vector<ScanlineHit>(sorted.begin() + i, sorted.begin() + j + 1);
}

inline void apply_scanline_hit_filters(std::vector<ScanlineHit> &hits, //
                                       const ScanlineHitFilterOptions &options)
{
    if (hits.empty()) {
        return;
    }
    if (options.max_height_offset > 0) {
        const double max_h = options.max_height_offset;
        hits.erase(std::remove_if(hits.begin(), hits.end(),
                                  [&](const ScanlineHit &h) {
                                      return std::fabs(h.vertical_offset) >
                                             max_h;
                                  }),
                   hits.end());
        if (hits.empty()) {
            return;
        }
    }
    if (!options.accept_opposite && options.min_dot > -1.0) {
        const double min_dot = options.min_dot;
        hits.erase(std::remove_if(
                       hits.begin(), hits.end(),
                       [&](const ScanlineHit &h) { return h.dot < min_dot; }),
                   hits.end());
        if (hits.empty()) {
            return;
        }
    } else if (options.accept_opposite && options.min_dot > 0.0) {
        const double pos_min = options.min_dot;
        const double neg_max = -options.min_dot;
        hits.erase(std::remove_if(hits.begin(), hits.end(),
                                  [&](const ScanlineHit &h) {
                                      return neg_max < h.dot && h.dot < pos_min;
                                  }),
                   hits.end());
        if (hits.empty()) {
            return;
        }
    }
    if (options.filter_out_hits_by_gap > 0) {
        filter_scanline_hits_by_gap(hits, options.filter_out_hits_by_gap);
        if (hits.empty()) {
            return;
        }
    }
    if (options.only_nearest) {
        sort_by_abs_lateral_offset(hits);
        hits.resize(1);
    }
}

struct FastCrossing
{
    using FlatBush = flatbush::FlatBush<double>;
    using PolylineType = RowVectors;
    using LabelsType = FlatBush::LabelsType;
    using IntersectionType = std::tuple<
        // intersection point, intersection ratio: t, s
        Eigen::Vector2d, Eigen::Vector2d,
        // (poly idx, pt idx), (poly idx, pt idx)
        Eigen::Vector2i, Eigen::Vector2i //
        >;

    FastCrossing(bool is_wgs84 = false) : is_wgs84_(is_wgs84) {}

    int add_polyline(const PolylineType &polyline, int index = -1)
    {
        bush_.reset();
        return quiver(polyline(0, 0), polyline(0, 1)).add(polyline, index);
    }

    int add_polyline(const Eigen::Ref<const FlatBush::PolylineType> &polyline,
                     int index = -1)
    {
        bush_.reset();
        return quiver(polyline(0, 0), polyline(0, 1)).add(polyline, index);
    }

    void finish() const
    {
        if (bush_) {
            return;
        }
        if (!quiver_) {
            return;
        }
        auto &polylines = quiver_->polylines();
        bush_ = FlatBush(polylines.size());
        for (auto &pair : polylines) {
            auto &idx = pair.first;
            auto &polyline = pair.second.polyline();
            bush_->Add(polyline.leftCols<2>(), idx);
        }
        bush_->Finish();
    }
    std::vector<IntersectionType> intersections() const
    {
        auto &bush = this->bush();
        auto boxes = bush.boxes();
        std::set<std::array<int, 2>> pairs;
        for (int i = 0, N = boxes.rows(); i < N; ++i) {
            const auto &box = boxes.row(i);
            double x0 = box[0];
            double y0 = box[1];
            double x1 = box[2];
            double y1 = box[3];
            auto hits = bush.Search(std::min(x0, x1), std::min(y0, y1),
                                    std::max(x0, x1), std::max(y0, y1));
            for (auto &h : hits) {
                int j = h;
                if (i < j) {
                    pairs.insert({i, j});
                }
            }
        }
        if (pairs.empty()) {
            return {};
        }
        auto labels = bush.labels();
        std::vector<IntersectionType> ret;
        for (auto &pair : pairs) {
            const auto &label1 = labels.row(pair[0]);
            const auto &label2 = labels.row(pair[1]);
            if (label1[0] == label2[0] && label1[1] + 1 == label2[1]) {
                continue;
            }
            const auto &seg1 = boxes.row(pair[0]);
            const auto &seg2 = boxes.row(pair[1]);
            auto intersect =
                intersect_segments((Eigen::Vector2d)seg1.head(2), //
                                   (Eigen::Vector2d)seg1.tail(2), //
                                   (Eigen::Vector2d)seg2.head(2), //
                                   (Eigen::Vector2d)seg2.tail(2));
            if (intersect) {
                auto &xy_st = *intersect;
                ret.emplace_back(
                    std::get<0>(xy_st),
                    Eigen::Vector2d(std::get<1>(xy_st), std::get<2>(xy_st)),
                    label1, label2);
            }
        }
        return ret;
    }

    std::vector<IntersectionType> intersections(
        const std::tuple<double, double> &z_offset_range,
        // 2: no check, 1: only self intersection, 0: no self intersection
        int self_intersection = 2) const
    {
        double z_min = std::get<0>(z_offset_range);
        double z_max = std::get<1>(z_offset_range);
        if (std::isinf(z_max)) {
            z_max = std::numeric_limits<double>::max();
        }
        if (z_min > z_max || z_max < 0) {
            return {};
        }

        auto v = this->intersections();
        // filter by intersection
        if (self_intersection == 0) {
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [](const auto &inter) {
                                       return std::get<2>(inter)[0] ==
                                              std::get<3>(inter)[0];
                                   }),
                    v.end());
        } else if (self_intersection == 1) {
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [](const auto &inter) {
                                       return std::get<2>(inter)[0] !=
                                              std::get<3>(inter)[0];
                                   }),
                    v.end());
        }

        // filter by z
        if (z_min > 0 || z_max < std::numeric_limits<double>::max()) {
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [&](auto &inter) { //
                                       auto p0 = this->coordinates(inter, true);
                                       auto p1 =
                                           this->coordinates(inter, false);
                                       double zz = std::fabs(p0[2] - p1[2]);
                                       return zz < z_min || zz > z_max;
                                   }),
                    v.end());
        }

        return v;
    }

    std::vector<IntersectionType> intersections(const Eigen::Vector2d &p0,
                                                const Eigen::Vector2d &p1,
                                                bool dedup = true) const
    {
        auto &bush = this->bush();
        double x0 = p0[0], y0 = p0[1];
        double x1 = p1[0], y1 = p1[1];
        auto hits = bush.Search(std::min(x0, x1), std::min(y0, y1),
                                std::max(x0, x1), std::max(y0, y1));
        if (hits.empty()) {
            return {};
        }
        std::vector<IntersectionType> ret;
        Eigen::Vector2i label1(0, 0);
        auto boxes = bush.boxes();
        auto labels = bush.labels();
        for (auto &idx : hits) {
            const auto &seg = boxes.row(idx);
            auto intersect = intersect_segments(p0, p1, //
                                                (Eigen::Vector2d)seg.head(2),
                                                (Eigen::Vector2d)seg.tail(2));
            if (!intersect) {
                continue;
            }
            const auto &label2 = labels.row(idx);
            auto &xy_ts = *intersect;
            ret.emplace_back(
                std::get<0>(xy_ts),
                Eigen::Vector2d(std::get<1>(xy_ts), std::get<2>(xy_ts)), //
                label1, label2);
        }
        std::sort(ret.begin(), ret.end(), [](const auto &t1, const auto &t2) {
            double tt1 = std::get<1>(t1)[0];
            double tt2 = std::get<1>(t2)[0];
            if (tt1 != tt2) {
                return tt1 < tt2;
            }
            return std::make_tuple(std::get<3>(t1)[0], std::get<3>(t1)[1]) <
                   std::make_tuple(std::get<3>(t2)[0], std::get<3>(t2)[1]);
        });
        if (dedup) {
            auto last = std::unique(
                ret.begin(), ret.end(), [](const auto &t1, const auto &t2) {
                    // xy, ts, (poly_idx, seg_idx), (poly_idx, seg_idx)
                    if (std::get<3>(t1)[0] != std::get<3>(t2)[0]) {
                        return false;
                    }
                    return (std::get<3>(t1)[1] == std::get<3>(t2)[1] &&
                            std::get<1>(t1)[1] == std::get<1>(t2)[1]) ||
                           ((std::get<3>(t1)[1] + 1) == std::get<3>(t2)[1] &&
                            std::get<1>(t1)[1] == 1.0 &&
                            std::get<1>(t2)[1] == 0.0) ||
                           ((std::get<3>(t1)[1] - 1) == std::get<3>(t2)[1] &&
                            std::get<1>(t1)[1] == 0.0 &&
                            std::get<1>(t2)[1] == 1.0);
                });
            ret.erase(last, ret.end());
        }
        return ret;
    }

    // xy, ts, label_of_cur_line, label_of_tree_line
    std::vector<IntersectionType> intersections(const PolylineType &polyline,
                                                bool dedup = true) const
    {
        std::vector<IntersectionType> ret;
        int N = polyline.rows();
        for (int i = 0; i < N - 1; ++i) {
            auto hit =
                intersections((Eigen::Vector2d)polyline.row(i).head(2),     //
                              (Eigen::Vector2d)polyline.row(i + 1).head(2), //
                              dedup);
            if (hit.empty()) {
                continue;
            }
            for (auto &h : hit) {
                auto &label_of_curr_seg = std::get<2>(h);
                label_of_curr_seg[1] = i;
            }
            bool has_dup = false;
            if (dedup && !ret.empty()) {
                // xy, ts, label1, label2
                auto &prev = ret.back();
                auto &curr = hit.front();
                auto &prev_label = std::get<3>(prev);
                auto &curr_label = std::get<3>(curr);
                if ((prev_label == curr_label &&
                     std::get<1>(prev)[1] == std::get<1>(curr)[1]) ||
                    (prev_label[0] == curr_label[0] &&
                     (prev_label[1] + 1) == curr_label[1] &&
                     std::get<1>(prev)[1] == 1.0 &&
                     std::get<1>(curr)[1] == 0.0) ||
                    (prev_label[0] == curr_label[0] &&
                     (prev_label[1] - 1) == curr_label[1] &&
                     std::get<1>(prev)[1] == 0.0 &&
                     std::get<1>(curr)[1] == 1.0)) {
                    has_dup = true;
                }
            }
            ret.insert(ret.end(), hit.begin() + (has_dup ? 1 : 0), hit.end());
        }
        return ret;
    }

    std::vector<IntersectionType>
    intersections(const Eigen::Ref<const FlatBush::PolylineType> &polyline,
                  bool dedup = true) const
    {
        PolylineType Nx3(polyline.rows(), 3);
        Nx3.leftCols<2>() = polyline;
        Nx3.col(2).setZero();
        return intersections(Nx3, dedup);
    }

    static Eigen::Vector3d coordinates(const RowVectors &xyzs, int seg_index,
                                       double t)
    {
        return xyzs.row(seg_index) * (1.0 - t) + xyzs.row(seg_index + 1) * t;
    }
    Eigen::Vector3d coordinates(int polyline_index, int seg_index,
                                double t) const
    {
        const PolylineRuler *ruler =
            quiver_ ? quiver_->polyline(polyline_index) : nullptr;
        if (!ruler) {
            throw std::out_of_range("[exception stub] map::at " +
                                    std::to_string(polyline_index));
        }
        return coordinates(ruler->polyline(), seg_index, t);
    }
    Eigen::Vector3d coordinates(const Eigen::Vector2i &index, double t) const
    {
        return coordinates(index[0], index[1], t);
    }
    Eigen::Vector3d coordinates(const IntersectionType &idx,
                                bool second = true) const
    {
        auto &ts = std::get<1>(idx);
        auto &idx1 = std::get<2>(idx);
        auto &idx2 = std::get<3>(idx);
        if (second) {
            return coordinates(idx2, ts[1]);
        }
        return coordinates(idx1, ts[0]);
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> arrow(int polyline_index,
                                                      int point_index) const
    {
        auto polyline = quiver_->polyline(polyline_index);
        return {polyline->at(point_index), polyline->dir(point_index)};
    }

    std::vector<IntersectionType> intersections(const PolylineType &polyline,
                                                double z_min, double z_max,
                                                bool dedup = true) const
    {
        if (z_min > z_max) {
            return {};
        }
        auto ret = intersections(polyline, dedup);
        if (ret.empty()) {
            return {};
        }
        ret.erase(std::remove_if(
                      ret.begin(), ret.end(),
                      [&](auto &idx) {
                          auto &ts = std::get<1>(idx);
                          auto &idx1 = std::get<2>(idx);
                          auto &idx2 = std::get<3>(idx);
                          double z0 = coordinates(polyline, idx1[1], ts[0])[2];
                          double zz = coordinates(idx2[0], idx2[1], ts[1])[2];
                          return (zz < z0 + z_min) || (zz > z0 + z_max);
                      }),
                  ret.end());
        return ret;
    }

    std::vector<IntersectionType>
    intersections(const Eigen::Ref<const FlatBush::PolylineType> &polyline,
                  double z_min, double z_max, bool dedup = true) const
    {
        if (z_min > z_max) {
            return {};
        }
        return intersections(to_Nx3(polyline), z_min, z_max, dedup);
    }

    // segment index
    Eigen::Vector2i segment_index(int index) const
    {
        return bush_->label(index);
    }
    std::vector<Eigen::Vector2i>
    segment_index(const Eigen::VectorXi &indexes) const
    {
        const int N = indexes.size();
        std::vector<Eigen::Vector2i> ret;
        ret.reserve(N);
        for (int i = 0; i < N; ++i) {
            ret.push_back(segment_index(indexes[i]));
        }
        return ret;
    }
    // point index
    Eigen::Vector2i point_index(int index) const
    {
        return quiver_->index(index);
    }
    std::vector<Eigen::Vector2i>
    point_index(const Eigen::VectorXi &indexes) const
    {
        const int N = indexes.size();
        std::vector<Eigen::Vector2i> ret;
        ret.reserve(N);
        for (int i = 0; i < N; ++i) {
            ret.push_back(point_index(indexes[i]));
        }
        return ret;
    }
    int point_index(int polyline_index, int point_index) const
    {
        return quiver_->index(polyline_index, point_index);
    }

    std::vector<Eigen::Vector2i>
    within(const Eigen::Vector2d &min, const Eigen::Vector2d &max,
           bool segment_wise = true, // else point-wise
           bool sort = true) const
    {
        auto hits = bush().Search(min[0], min[1], max[0], max[1]);
        if (hits.empty()) {
            return {};
        }
        std::vector<Eigen::Vector2i> ret;
        if (segment_wise) {
            ret = segment_index(Eigen::VectorXi::Map(&hits[0], hits.size()));
        } else {
            auto points = std::unordered_set<Eigen::Vector2i,
                                             hash_eigen<Eigen::Vector2i>>{};
            for (auto &idx : hits) {
                auto index = segment_index(idx);
                // first point of segment
                points.insert(index);
                index[1] += 1;
                // second point of segment
                points.insert(index);
            }
            for (auto &idx : points) {
                auto &xyzs = quiver_->polyline(idx[0])->polyline();
                double x = xyzs(idx[1], 0);
                if (x < min[0] || x > max[0]) {
                    continue;
                }
                double y = xyzs(idx[1], 1);
                if (y < min[1] || y > max[3]) {
                    continue;
                }
                ret.push_back(idx);
            }
        }
        if (sort) {
            sort_indexes(ret);
        }
        return ret;
    }
    std::vector<Eigen::Vector2i>
    within(const Eigen::Ref<const RowVectorsNx2> &polygon,
           bool segment_wise = true, //
           bool sort = true) const
    {
        Eigen::Vector2d min = polygon.colwise().minCoeff();
        Eigen::Vector2d max = polygon.colwise().maxCoeff();
        auto hits = bush().Search(min[0], min[1], max[0], max[1]);
        if (hits.empty()) {
            return {};
        }
        const int N = hits.size();
        std::vector<Eigen::Vector2i> segs;
        segs.reserve(N);
        RowVectorsNx2 P0(N, 2);
        RowVectorsNx2 P1(N, 2);
        {
            int i = -1;
            for (auto &idx : hits) {
                ++i;
                auto index = segment_index(idx);
                segs.push_back(index);
                auto &xyzs = quiver_->polyline(index[0])->polyline();
                P0.row(i) = xyzs.row(index[1]).head(2);
                P1.row(i) = xyzs.row(index[1] + 1).head(2);
            }
        }
        auto mask0 = point_in_polygon(P0, polygon);
        auto mask1 = point_in_polygon(P1, polygon);

        auto indexes =
            std::unordered_set<Eigen::Vector2i, hash_eigen<Eigen::Vector2i>>{};
        if (segment_wise) {
            // intersects polygon itself
            for (auto &xy_ts_label1_label2 : intersections(to_Nx3(polygon))) {
                indexes.insert(std::get<3>(xy_ts_label1_label2));
            }
            // one point of segment inside polygon: mask0 | mask1
            for (int i = 0; i < N; ++i) {
                if (mask0[i] || mask1[i]) {
                    indexes.insert(segs[i]);
                }
            }
        } else {
            for (int i = 0; i < N; ++i) {
                if (mask0[i]) {
                    indexes.insert(segs[i]);
                }
            }
            for (int i = 0; i < N; ++i) {
                if (mask1[i]) {
                    indexes.insert({segs[i][0], segs[i][1] + 1});
                }
            }
        }
        auto ret = std::vector<Eigen::Vector2i>(indexes.begin(), indexes.end());
        if (sort) {
            sort_indexes(ret);
        }
        return ret;
    }

    std::vector<Eigen::Vector2i> within(const Eigen::Vector2d &center,
                                        double width, double height,
                                        double heading = 0.0,
                                        bool segment_wise = true, //
                                        bool sort = true) const
    {
        double rad = (90.0 - heading) / 180.0 * PI;
        Eigen::Vector2d y(std::cos(rad), std::sin(rad));
        Eigen::Vector2d x(y[1], -y[0]);
        RowVectorsNx2 polygon(5, 3);
        polygon.row(0) = width / 2 * x - height / 2 * y;
        polygon.row(1) = width / 2 * x + height / 2 * y;
        polygon.row(2) = -width / 2 * x + height / 2 * y;
        polygon.row(3) = -width / 2 * x - height / 2 * y;
        polygon.row(4) = width / 2 * x - height / 2 * y;
        if (is_wgs84_) {
            polygon = enu2lla(to_Nx3(polygon),
                              Eigen::Vector3d(center[0], center[1], 0.0))
                          .leftCols(2);
        } else {
            polygon.col(0).array() += center[0];
            polygon.col(1).array() += center[1];
        }
        return within(polygon, segment_wise, sort);
    }

    // nearest
    std::pair<Eigen::Vector2i, double>
    nearest(const Eigen::Vector3d &position, //
            bool return_squared_l2 = false) const
    {
        auto [idx, dist] = quiver_->nearest(position, return_squared_l2);
        return {point_index(idx), dist};
    }
    std::pair<Eigen::Vector2i, double>
    nearest(int index, //
            bool return_squared_l2 = false) const
    {
        auto [idx, dist] = quiver_->nearest(index, return_squared_l2);
        return {point_index(idx), dist};
    }
    std::pair<Eigen::Vector2i, double>
    nearest(const Eigen::Vector2i &index, //
            bool return_squared_l2 = false) const
    {
        return nearest(point_index(index[0], index[1]), return_squared_l2);
    }

    std::pair<IntNx2, Eigen::VectorXd> nearest(
        const Eigen::Vector3d &position, //
        std::optional<int> k = std::nullopt,
        std::optional<double> radius = std::nullopt,
        bool sort = true, //
        bool return_squared_l2 = false,
        std::optional<std::pair<Eigen::Vector3d, Quiver::FilterParams>> filter =
            std::nullopt) const
    {
        if (!k && !radius) {
            throw std::invalid_argument("should specify k or radius");
        }
        auto [ii, dd] =
            k ? quiver_->nearest(position, *k, sort, return_squared_l2)
              : quiver_->nearest(position, *radius, sort, return_squared_l2);
        if (filter) {
            auto [dir, params] = *filter;
            auto ii_dd = quiver_->filter(ii, dd, Arrow(position, dir), params);
            ii = ii_dd.first;
            dd = ii_dd.second;
        }
        auto vec_ii = point_index(ii);
        return {Eigen::Map<const IntNx2>(vec_ii[0].data(), vec_ii.size(), 2),
                dd};
    }

    // --- batch queries -------------------------------------------------
    //
    // Both `nearest_batch` and `intersections_batch` fix the miss/truncation
    // semantics up front, instead of leaving each caller to invent its own:
    //
    //   * hit count    : `counts[i]` is always the *real* number of hits
    //                    after filtering, before any `max_hits` truncation.
    //                    So `counts[i] > max_hits` means "truncated".
    //   * miss         : label = -1, distance = inf, and for
    //                    `intersections_batch` lateral/seg/t/dot/lla are
    //                    inf/NaN free: lateral = inf, t = NaN, dot = NaN,
    //                    lla = (NaN, NaN, NaN), dir = (NaN, NaN, 0).
    //   * truncation   : keep the first `max_hits` candidates in
    //                    |lateral| ascending order (ties broken by
    //                    (polyline_index, segment_index)), which is exactly
    //                    the candidate order a per-point
    //                    `min(hits, key=lambda h: abs(h.lateral_offset))`
    //                    would have seen after sorting.
    //   * empty index  : width is still `max_hits` (never 0), so numpy shapes
    //                    stay predictable; every point misses.

    // Batch `nearest`, one C++ loop over all queries.
    //
    // Exactly one of `k` / `radius` should be given; none of them means
    // `k=1` (nearest point).
    //
    // Returns (counts, polyline_index, point_index, dists):
    //   counts         : (N,)   real hit count per query (before truncation)
    //   polyline_index : (N, H) hit polyline index, -1 for unused slots
    //   point_index    : (N, H) hit point index inside that polyline, -1 unused
    //   dists          : (N, H) distance, inf for unused slots
    // H = min(k, #points) for `k` mode, or `max_hits` / the widest hit count
    // for `radius` mode. Hits are sorted by distance and truncated before
    // padding, so `[i, :counts[i]]` is exactly what a per-point
    // `nearest(..., k=..., radius=...)` returns.
    //
    // Note this returns the nearest *points*, not the nearest *segments* (the
    // single point `nearest` is point based too). For a segment-wise candidate
    // search use `intersections_batch`, which works on segments and is the one
    // that matches the "scanline candidates" use case.
    using BatchNearest = std::tuple<Eigen::VectorXi, // counts (N,)
                                    Eigen::MatrixXi, // polyline_index (N, H)
                                    Eigen::MatrixXi, // point_index (N, H)
                                    Eigen::MatrixXd  // dists (N, H)
                                    >;
    BatchNearest nearest_batch(const RowVectors &positions, //
                               std::optional<int> k = std::nullopt,
                               std::optional<double> radius = std::nullopt,
                               std::optional<int> max_hits = std::nullopt,
                               bool return_squared_l2 = false) const
    {
        if (!k && !radius) {
            k = 1; // default: nearest point
        }
        if (k.has_value() && radius.has_value()) {
            throw std::invalid_argument(
                "specify exactly one of k / radius for nearest_batch");
        }
        if (k && *k <= 0) {
            throw std::invalid_argument("k should be positive");
        }
        const int N = positions.rows();
        if (!quiver_) {
            const int H = max_hits ? *max_hits : (k ? *k : 0);
            return {Eigen::VectorXi::Zero(N),
                    Eigen::MatrixXi::Constant(N, H, -1),
                    Eigen::MatrixXi::Constant(N, H, -1),
                    Eigen::MatrixXd::Constant(
                        N, H, std::numeric_limits<double>::infinity())};
        }
        return quiver_->nearest_batch(positions, k, radius, max_hits,
                                      return_squared_l2);
    }

    // Batch scanline search: "for each of N points, give me the polyline
    // segments crossing a perpendicular scanline through that point".
    //
    // This is the batch form of the per-point loop
    //     hits = fc.intersections(lla=p, dir=d, min=-r, max=r, options=o)
    // with identical filtering and ordering semantics.
    //
    // `llas` : (N, 3), ENU or WGS84 depending on `is_wgs84()`
    // `dirs` : (N, 3), unit vectors. A row containing a NaN means "no
    //          direction for this query": the scanline is built from (1, 0, 0)
    //          (a pure north-south scanline) and the direction filters
    //          (`min_dot` / `accept_opposite`) are skipped for that query.
    //
    // Returns (counts, lateral, polyline_id, seg_idx, t, dot, lla, dir):
    //   counts      : (N,)     real hit count, *before* `max_hits` truncation
    //   lateral     : (N, H)   signed lateral offset within [min, max], left
    //                          positive, |lateral| ascending inside a row
    //   polyline_id : (N, H)   hit polyline index, -1 unused
    //   seg_idx     : (N, H)   segment index inside that polyline, -1 unused
    //   t           : (N, H)   parameter on that segment, NaN unused
    //   dot         : (N, H)   dot(query dir, hit segment dir), NaN unused
    //   lla         : (N*H, 3) hit point, NaN unused (reshape to (N, H, 3))
    //   dir         : (N*H, 3) hit segment direction, NaN unused
    // H = max_hits, or the widest hit count of the batch when not given.
    using BatchScanlineHits = std::tuple<Eigen::VectorXi, // counts
                                         Eigen::MatrixXd, // lateral
                                         Eigen::MatrixXi, // polyline_id
                                         Eigen::MatrixXi, // seg_idx
                                         Eigen::MatrixXd, // t
                                         Eigen::MatrixXd, // dot
                                         Eigen::MatrixXd, // lla (N*H, 3)
                                         Eigen::MatrixXd  // dir (N*H, 3)
                                         >;
    BatchScanlineHits
    intersections_batch(const RowVectors &llas, const RowVectors &dirs, //
                        double min = -5.0, double max = 5.0,
                        const ScanlineHitFilterOptions &options = {},
                        std::optional<int> max_hits = std::nullopt) const
    {
        const int N = llas.rows();
        if (dirs.rows() != N) {
            throw std::invalid_argument("#llas != #dirs");
        }
        if (min > max) {
            throw std::invalid_argument("min should be <= max");
        }
        if (max_hits && *max_hits < 0) {
            throw std::invalid_argument("max_hits should be non-negative");
        }

        std::vector<std::vector<ScanlineHit>> hits_per_query(N);
        int width = max_hits ? *max_hits : 0;
        for (int i = 0; i < N; ++i) {
            hits_per_query[i] =
                scanline_hits(Eigen::Vector3d(llas.row(i)),
                              Eigen::Vector3d(dirs.row(i)), min, max, options);
            if (!max_hits) {
                width =
                    std::max(width, static_cast<int>(hits_per_query[i].size()));
            }
        }
        const int H = width;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();

        Eigen::VectorXi counts(N);
        Eigen::MatrixXd lateral = Eigen::MatrixXd::Constant(N, H, inf);
        Eigen::MatrixXi polyline_id = Eigen::MatrixXi::Constant(N, H, -1);
        Eigen::MatrixXi seg_idx = Eigen::MatrixXi::Constant(N, H, -1);
        Eigen::MatrixXd t = Eigen::MatrixXd::Constant(N, H, nan);
        Eigen::MatrixXd dot = Eigen::MatrixXd::Constant(N, H, nan);
        Eigen::MatrixXd lla = Eigen::MatrixXd::Constant(N * H, 3, nan);
        Eigen::MatrixXd dir = Eigen::MatrixXd::Constant(N * H, 3, nan);
        for (int i = 0; i < N; ++i) {
            auto &hits = hits_per_query[i];
            counts[i] = static_cast<int>(hits.size());
            sort_by_abs_lateral_offset(hits);
            const int m = std::min(H, static_cast<int>(hits.size()));
            for (int j = 0; j < m; ++j) {
                const auto &h = hits[j];
                lateral(i, j) = h.lateral_offset;
                polyline_id(i, j) = h.polyline_index;
                seg_idx(i, j) = h.segment_index;
                t(i, j) = h.t;
                dot(i, j) = h.dot;
                lla.row(i * H + j) = h.lla.transpose();
                dir.row(i * H + j) = h.dir.transpose();
            }
        }
        return {counts, lateral, polyline_id, seg_idx, t, dot, lla, dir};
    }

    // Single scanline query, the exact code path `intersections_batch` uses
    // (exposed for per-point control, and for checking the batch result).
    //
    // This is *not* the legacy segment/segment `intersections` family. It
    // places a scanline perpendicular to `dir` through `lla`, searches
    // [min, max] laterally and returns every segment crossing it, filtered by
    // `options` and sorted by |lateral|.
    std::vector<ScanlineHit>
    scanline_hits(const Eigen::Vector3d &lla, const Eigen::Vector3d &dir, //
                  double min = -5.0, double max = 5.0,
                  const ScanlineHitFilterOptions &options = {}) const
    {
        if (!quiver_) {
            return {};
        }
        auto hits = scanline_hits(lla, dir, min, max, options, is_wgs84_);
        // same deterministic order as `intersections_batch`
        sort_by_abs_lateral_offset(hits);
        return hits;
    }

    bool is_wgs84() const { return is_wgs84_; }
    int num_polylines() const
    {
        return quiver_ ? quiver_->polylines().size() : 0;
    }
    const std::map<int, PolylineRuler> &polyline_rulers() const
    {
        static const std::map<int, PolylineRuler> dummy;
        if (!quiver_) {
            return dummy;
        }
        return quiver_->polylines();
    }
    const PolylineRuler *polyline_ruler(int label) const
    {
        if (!quiver_) {
            return nullptr;
        }
        return quiver_->polyline(label);
    }

    const FlatBush *export_bush(bool autobuild = true) const
    {
        if (autobuild) {
            finish();
        }
        return bush_ ? &*bush_ : nullptr;
    }
    const KdQuiver *export_quiver() const
    {
        return quiver_ ? &*quiver_ : nullptr;
    }

  private:
    const bool is_wgs84_{false};

    // Core of the perpendicular scanline query, shared by `scanline_hits` and
    // `intersections_batch`.
    //
    // The scanline goes through `lla`, perpendicular to `dir`, and spans the
    // lateral offsets [min, max] along the *leftward* normal of `dir` (so
    // min=-r/max=r is symmetric). Coordinates are intersected in the frame the
    // polylines were added in (ENU or WGS84); for WGS84 the offsets are scaled
    // by the cheap-ruler `k` of the query latitude first, so `min`/`max` stay
    // in meters.
    //
    // Candidate search uses the segment R-tree (`bush_`, segment bounding
    // boxes), so a segment is visited at most once per query and only the
    // segments whose bbox overlaps the scanline bbox get `intersect_segments`:
    // with `radius = 1.2m` this touches a handful of segments regardless of how
    // many points the polylines have.
    std::vector<ScanlineHit>
    scanline_hits(const Eigen::Vector3d &lla, const Eigen::Vector3d &dir, //
                  double min, double max,
                  const ScanlineHitFilterOptions &options, bool is_wgs84) const
    {
        std::vector<ScanlineHit> hits;
        if (!quiver_) {
            return hits;
        }
        const double nan = std::numeric_limits<double>::quiet_NaN();
        bool has_dir = !std::isnan(dir[0]) && !std::isnan(dir[1]);
        Eigen::Vector3d query_dir(1.0, 0.0, 0.0);
        if (has_dir) {
            Eigen::Vector2d d(dir[0], dir[1]);
            double norm = d.norm();
            if (norm > 0.0) {
                query_dir = Eigen::Vector3d(d[0] / norm, d[1] / norm, 0.0);
            } else {
                has_dir = false;
            }
        }
        // leftward normal of the query direction (0 -> +pi/2), metric space
        Eigen::Vector3d left(-query_dir[1], query_dir[0], 0.0);

        // scanline endpoints in the frame of the polylines
        Eigen::Vector3d src = lla + left * min;
        Eigen::Vector3d dst = lla + left * max;
        if (is_wgs84) {
            Eigen::Vector3d k = Quiver::k(lla[1]);
            // metric offset -> angular offset
            for (int i = 0; i < 3; ++i) {
                src[i] = lla[i] + left[i] * min / k[i];
                dst[i] = lla[i] + left[i] * max / k[i];
            }
        }

        const auto &bush = this->bush();
        const auto labels = bush.labels();
        const auto boxes = bush.boxes();
        double x0 = src[0], y0 = src[1];
        double x1 = dst[0], y1 = dst[1];
        auto candidates = bush.Search(std::min(x0, x1), std::min(y0, y1),
                                      std::max(x0, x1), std::max(y0, y1));
        if (candidates.empty()) {
            return hits;
        }

        hits.reserve(candidates.size());
        const double lateral_span = max - min;
        const auto &rulers = quiver_->polylines();
        const Eigen::Vector2d p0(src[0], src[1]);
        const Eigen::Vector2d p1(dst[0], dst[1]);
        for (int idx : candidates) {
            const auto &label = labels.row(idx);
            const auto &seg = boxes.row(idx);
            auto intersect =
                intersect_segments(p0, p1, (Eigen::Vector2d)seg.head(2),
                                   (Eigen::Vector2d)seg.tail(2));
            if (!intersect) {
                continue;
            }
            const auto &xy_st = *intersect;
            const double t_scan = std::get<1>(xy_st);
            const int polyline_index = label[0];
            const int segment_index = label[1];
            auto itr = rulers.find(polyline_index);
            if (itr == rulers.end()) {
                continue;
            }
            const auto &ruler = itr->second;
            ScanlineHit h;
            h.polyline_index = polyline_index;
            h.segment_index = segment_index;
            h.t = std::get<2>(xy_st);
            h.lla = ruler.at(segment_index, h.t);
            h.dir = ruler.dir(segment_index);
            h.lateral_offset = min + lateral_span * t_scan;
            h.vertical_offset = h.lla[2] - lla[2];
            h.dot = has_dir ? query_dir.dot(h.dir) : nan;
            hits.push_back(h);
        }

        if (has_dir) {
            apply_scanline_hit_filters(hits, options);
        } else {
            // no direction given: only the direction gates are skipped
            auto opts = options;
            opts.accept_opposite = false;
            opts.min_dot = -2.0;
            apply_scanline_hit_filters(hits, opts);
        }
        return hits;
    }

    std::unique_ptr<KdQuiver> quiver_;
    KdQuiver &quiver(double lon, double lat)
    {
        if (!quiver_) {
            quiver_ =
                is_wgs84_
                    ? std::make_unique<KdQuiver>(Eigen::Vector3d(lon, lat, 0.0))
                    : std::make_unique<KdQuiver>();
        }
        return *quiver_;
    }

    // auto rebuild flatbush
    mutable std::optional<FlatBush> bush_;
    const FlatBush &bush() const
    {
        finish();
        return *bush_;
    }
};
} // namespace cubao

#endif
