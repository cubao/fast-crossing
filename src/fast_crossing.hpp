#ifndef CUBAO_FAST_CROSSING_HPP
#define CUBAO_FAST_CROSSING_HPP

#include "polyline_ruler.hpp"
#include "flatbush.h"
#include <set>
#include <vector>

namespace cubao
{
struct FastCrossing
{
    using FlatBush = flatbush::FlatBush<double>;
    using PolylineType = FlatBush::PolylineType;
    using LabelsType = FlatBush::LabelsType;
    using IntersectionType = std::tuple<
        // intersection point, intersection ratio: t, s
        Eigen::Vector2d, Eigen::Vector2d,
        // (poly idx, pt idx), (poly idx, pt idx)
        Eigen::Vector2i, Eigen::Vector2i //
        >;

    void add_polyline(const Eigen::Ref<const PolylineType> &polyline,
                      int index = -1)
    {
        if (index < 0) {
            index = num_poylines_;
        }
        ++num_poylines_;
        bush_.Add(polyline, index);
    }
    void finish() { bush_.Finish(); }
    std::vector<IntersectionType> intersections() const
    {
        auto boxes = bush_.boxes();
        std::set<std::array<int, 2>> pairs;
        for (int i = 0, N = boxes.rows(); i < N; ++i) {
            const auto &box = boxes.row(i);
            double x0 = box[0];
            double y0 = box[1];
            double x1 = box[2];
            double y1 = box[3];
            auto hits = bush_.Search(std::min(x0, x1), std::min(y0, y1),
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
        auto labels = bush_.labels();
        std::vector<IntersectionType> ret;
        for (auto &pair : pairs) {
            const auto &label1 = labels.row(pair[0]);
            const auto &label2 = labels.row(pair[1]);
            if (label1[0] == label2[0] && label1[1] + 1 == label2[1]) {
                continue;
            }
            const auto &seg1 = boxes.row(pair[0]);
            const auto &seg2 = boxes.row(pair[1]);
            auto intersect = intersect_segments(
                (Eigen::Vector2d)seg1.head(2), (Eigen::Vector2d)seg1.tail(2), //
                (Eigen::Vector2d)seg2.head(2), (Eigen::Vector2d)seg2.tail(2));
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
        double x0 = p0[0], y0 = p0[1];
        double x1 = p1[0], y1 = p1[1];
        auto hits = bush_.Search(std::min(x0, x1), std::min(y0, y1),
                                 std::max(x0, x1), std::max(y0, y1));
        if (hits.empty()) {
            return {};
        }
        std::vector<IntersectionType> ret;
        Eigen::Vector2i label1(0, 0);
        auto boxes = bush_.boxes();
        auto labels = bush_.labels();
        for (auto &idx : hits) {
            const auto &seg = boxes.row(idx);
            auto intersect = intersect_segments(
                (Eigen::Vector2d)p0, (Eigen::Vector2d)p1,
                (Eigen::Vector2d)seg.head(2), (Eigen::Vector2d)seg.tail(2));
            if (!intersect) {
                continue;
            }
            const auto &label2 = labels.row(idx);
            auto &xy_ts = *intersect;
            ret.emplace_back(
                std::get<0>(xy_ts),
                Eigen::Vector2d(std::get<1>(xy_ts), std::get<2>(xy_ts)), label1,
                label2);
        }
        std::sort(ret.begin(), ret.end(), [](const auto &t1, const auto &t2) {
            return std::get<1>(t1)[0] < std::get<1>(t2)[0];
        });
        if (dedup) {
            auto last = std::unique(
                ret.begin(), ret.end(), [](const auto &t1, const auto &t2) {
                    return std::get<0>(t1) == std::get<0>(t2);
                });
            ret.erase(last, ret.end());
        }
        return ret;
    }

    std::vector<IntersectionType>
    intersections(const Eigen::Ref<const PolylineType> &polyline,
                  bool dedup = true) const
    {
        std::vector<IntersectionType> ret;
        int N = polyline.rows();
        for (int i = 0; i < N - 1; ++i) {
            auto hit =
                intersections(polyline.row(i), polyline.row(i + 1), dedup);
            if (hit.empty()) {
                continue;
            }
            for (auto &h : hit) {
                auto &label_of_curr_seg = std::get<2>(h);
                label_of_curr_seg[1] = i;
            }
            if (dedup && !ret.empty()) {
                // xy, ts, label1, label2
                auto &prev = ret.back();
                auto &curr = hit.front();
                if (std::get<0>(prev) == std::get<0>(curr)) {
                    ret.pop_back();
                }
            }
            ret.insert(ret.end(), hit.begin(), hit.end());
        }
        return ret;
    }

    const FlatBush &bush() const { return bush_; }
    int num_poylines() const { return num_poylines_; }

  private:
    FlatBush bush_;
    int num_poylines_ = 0;
};
} // namespace cubao

#endif
