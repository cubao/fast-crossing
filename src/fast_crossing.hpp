#ifndef CUBAO_FAST_CROSSING_HPP
#define CUBAO_FAST_CROSSING_HPP

#include "polyline_ruler.hpp"
#include "flatbush.h"
#include <set>
#include <vector>
#include <optional>

namespace cubao
{
struct FastCrossing
{
    using FlatBush = flatbush::FlatBush<double>;
    using PolylineType =
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
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
        if (index < 0) {
            index = polyline_rulers_.size();
        }
        if (polyline_rulers_.find(index) != polyline_rulers_.end()) {
            throw std::invalid_argument("duplicate index: " +
                                        std::to_string(index));
        }
        polyline_rulers_.emplace(index, PolylineRuler(polyline, is_wgs84_));
        bush_.reset();
        return index;
    }

    int add_polyline(const Eigen::Ref<const FlatBush::PolylineType> &polyline,
                     int index = -1)
    {
        PolylineType Nx3(polyline.rows(), 3);
        Nx3.leftCols<2>() = polyline;
        Nx3.col(2).setZero();
        return add_polyline(Nx3, index);
    }

    void finish() const
    {
        if (bush_) {
            return;
        }
        bush_ = FlatBush(polyline_rulers_.size());
        for (auto &pair : polyline_rulers_) {
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
        auto &ruler = polyline_rulers_.at(polyline_index);
        return coordinates(ruler.polyline(), seg_index, t);
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

    const FlatBush &bush() const
    {
        finish();
        return *bush_;
    }
    bool is_wgs84() const { return is_wgs84_; }
    int num_poylines() const { return polyline_rulers_.size(); }
    const std::map<int, PolylineRuler> &polyline_rulers() const
    {
        return polyline_rulers_;
    }
    const PolylineRuler *polyline_ruler(int label) const
    {
        auto itr = polyline_rulers_.find(label);
        if (itr == polyline_rulers_.end()) {
            return nullptr;
        }
        return &itr->second;
    }

  private:
    const bool is_wgs84_{false};
    std::map<int, PolylineRuler> polyline_rulers_;
    // auto rebuild flatbush
    mutable std::optional<FlatBush> bush_;
};
} // namespace cubao

#endif
