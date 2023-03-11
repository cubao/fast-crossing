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

namespace cubao
{
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
            bush_ = FlatBush();
            bush_->Finish();
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
    // point index
    Eigen::Vector2i point_index(int index) const
    {
        return quiver_->index(index);
    }

    std::vector<Eigen::Vector2i>
    within(const Eigen::Vector4d &bbox,
           bool segment_wise = true /* else point-wise */)
    {
        std::vector<Eigen::Vector2i> ret;
        auto hits = bush().Search(bbox[0], bbox[1], bbox[2], bbox[3]);
        if (segment_wise) {
            ret.reserve(hits.size());
            for (auto &idx : hits) {
                ret.push_back(segment_index(idx));
            }
            return ret;
        }
        std::unordered_set<Eigen::Vector2i, hash_eigen<Eigen::Vector2i>> points;
        for (auto &idx : hits) {
            auto index = segment_index(idx);
            // first point of segment
            points.insert(index);
            index[1] += 1;
            // second point of segment
            points.insert(index);
        }
        return {};
    }
    std::vector<Eigen::Vector2i>
    within(const Eigen::Ref<const RowVectorsNx2> &polygon,
           bool segment_wise = true)
    {
        return {};
    }

    // nearest
    std::pair<int, double> nearest(const Eigen::Vector3d &position, //
                                   bool return_squared_l2 = false) const
    {
        return quiver_->nearest(position, return_squared_l2);
    }
    std::pair<int, double> nearest(int index, //
                                   bool return_squared_l2 = false) const
    {
        return quiver_->nearest(index, return_squared_l2);
    }
    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    nearest(const Eigen::Vector3d &position, //
            std::optional<int> k = std::nullopt,
            std::optional<double> radius = std::nullopt,
            bool sorted = true, //
            bool return_squared_l2 = false) const
    {
        return {};
    }

    const FlatBush &bush() const
    {
        finish();
        return *bush_;
    }
    bool is_wgs84() const { return is_wgs84_; }
    int num_poylines() const
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

  private:
    const bool is_wgs84_{false};

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
};
} // namespace cubao

#endif
