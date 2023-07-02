// should sync
// -
// https://github.com/cubao/fast-crossing/blob/master/src/polyline_in_polygon.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/polyline_in_polygon.hpp

#ifndef CUBAO_POLYLINE_IN_POLYGON
#define CUBAO_POLYLINE_IN_POLYGON

#include "fast_crossing.hpp"
#include "point_in_polygon.hpp"

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
    // pt, (t, s), cur_label=(poly1, seg1), tree_label=(poly2, seg2)
    auto ruler = PolylineRuler(polyline, fc.is_wgs84());
    // 0.0, [r1, r2, ..., length]
    const int N = intersections.size() + 1;
    Eigen::VectorXd ranges(N);
    int idx = -1;
    for (auto &inter : intersections) {
        int seg_idx = std::get<2>(inter)[1];
        double t = std::get<1>(inter)[0];
        double r = ruler.range(seg_idx, t);
        ranges[++idx] = r;
    }
    ranges[++idx] = ruler.length();
    RowVectorsNx2 midpoints(N, 3);
    {
        idx = 0;
        double r = 0.0;
        while (idx < N) {
            double rr = ranges[idx];
            midpoints.row(idx) = ruler.at((r + rr) / 2.0).head(2);
            r = rr;
            ++idx;
        }
    }
    auto mask = point_in_polygon(midpoints, polygon);
    PolylineChunks ret;
    {
        idx = 0;
        double r = 0.0;
        while (idx < N) {
            if (mask[idx] && ranges[idx] > r) {
                auto [seg1, t1] = ruler.segment_index_t(r);
                auto [seg2, t2] = ruler.segment_index_t(ranges[idx]);
                ret.emplace(std::make_tuple(seg1, t1, r, seg2, t2, ranges[idx]),
                            ruler.lineSliceAlong(r, ranges[idx]));
            }
            r = ranges[idx];
            ++idx;
        }
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
