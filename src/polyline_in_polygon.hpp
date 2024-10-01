// should sync
// -
// https://github.com/cubao/fast-crossing/blob/master/src/polyline_in_polygon.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/polyline_in_polygon.hpp

#ifndef CUBAO_POLYLINE_IN_POLYGON
#define CUBAO_POLYLINE_IN_POLYGON

#include "fast_crossing.hpp"
#include "point_in_polygon.hpp"
#include "dbg.h"

namespace cubao
{
using PolylineChunks = std::map<std::tuple<int,    // seg_idx
                                           double, // t
                                           double, // range
                                           int,    // seg_idx,
                                           double, // t
                                           double  // range
                                           >,
                                RowVectors>;
inline PolylineChunks
polyline_in_polygon(const RowVectors &polyline, //
                    const Eigen::Ref<const RowVectorsNx2> &polygon,
                    const FastCrossing &fc)
{
    auto intersections = fc.intersections(polyline);
    auto ruler = PolylineRuler(polyline, fc.is_wgs84());
    if (intersections.empty()) {
        int inside = point_in_polygon(polyline.block(0, 0, 1, 2), polygon)[0];
        if (!inside) {
            return {};
        }
        return {{0, 0.0, 0.0, ruler.N() - 2, 1.0, ruler.length()}, polyline};
    }
    // pt, (t, s), cur_label=(poly1, seg1), tree_label=(poly2, seg2)
    const int N = intersections.size() + 2;
    // init ranges
    std::vector<double> ranges;
    Eigen::VectorXd ranges(N);
    ranges.reserve(N);
    ranges.push_back(0.0);
    for (auto &inter : intersections) {
        int seg_idx = std::get<2>(inter)[1];
        double t = std::get<1>(inter)[0];
        double r = ruler.range(seg_idx, t);
        ranges.push_back(r);
    }
    ranges.push_back(ruler.length());
    // ranges o------o--------o-----------------o
    // midpts     ^       ^             ^
    std::vector<Eigen::Vector2d> midpoints;
    midpoints.reserve(ranges.size() - 1);
    for (int i = 0; i < N - 1; ++i) {
        double rr = (ranges[i] + ranges[i + 1]) / 2.0;
        midpoints.push_back(ruler.at(rr).head(2));
    }
    auto mask = point_in_polygon(Eigen::Map<const RowVectorsNx2>(
                                     midpoints[0].data(), midpoints.size(), 2),
                                 polygon);
    PolylineChunks ret;
    {
        /*
        double r = 0.0;
        while (idx < N) {
            // mask && length
            if (mask[idx] && ranges[idx] > r) {
                auto [seg1, t1] = ruler.segment_index_t(r);
                auto [seg2, t2] = ruler.segment_index_t(ranges[idx]);
                ret.emplace(std::make_tuple(seg1, t1, r, seg2, t2, ranges[idx]),
                            ruler.lineSliceAlong(r, ranges[idx]));
            }
            r = ranges[idx];
            ++idx;
        }
        */
    }
    return ret;
}

inline PolylineChunks
polyline_in_polygon(const RowVectors &polyline, //
                    const Eigen::Ref<const RowVectorsNx2> &polygon,
                    bool is_wgs84 = false)
{
    auto fc = FastCrossing(is_wgs84);
    fc.add_polyline(polygon);
    fc.finish();
    return polyline_in_polygon(polyline, polygon, fc);
}

} // namespace cubao

#endif
