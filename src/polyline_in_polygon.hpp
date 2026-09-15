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
using PolylineChunkLabel = std::tuple<int,    // seg_idx
                                      double, // t
                                      double, // range
                                      int,    // seg_idx
                                      double, // t
                                      double  // range
                                      >;
using PolylineChunkLabels = std::vector<PolylineChunkLabel>;
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
        return PolylineChunks{
            {{0, 0.0, 0.0, ruler.N() - 2, 1.0, ruler.length()}, polyline}};
    }
    // pt, (t, s), cur_label=(poly1, seg1), tree_label=(poly2, seg2)
    const int N = intersections.size() + 2;
    // init ranges
    Eigen::VectorXd ranges(N);
    {
        int idx = -1;
        ranges[++idx] = 0.0;
        for (auto &inter : intersections) {
            int seg_idx = std::get<2>(inter)[1];
            double t = std::get<1>(inter)[0];
            double r = ruler.range(seg_idx, t);
            ranges[++idx] = r;
        }
        ranges[++idx] = ruler.length();
    }
    // ranges o------o--------o-----------------o
    // midpts     ^       ^             ^
    RowVectorsNx2 midpoints(N - 1, 2);
    for (int i = 0; i < N - 1; ++i) {
        double rr = (ranges[i] + ranges[i + 1]) / 2.0;
        midpoints.row(i) = ruler.along(rr).head(2);
    }
    auto mask = point_in_polygon(midpoints, polygon);
    PolylineChunks ret;
    {
        for (int i = 0; i < N - 1; ++i) {
            double r1 = ranges[i];
            double r2 = ranges[i + 1];
            if (r2 <= r1 || mask[i] == 0) {
                continue;
            }
            auto [seg1, t1] = ruler.segment_index_t(r1);
            auto [seg2, t2] = ruler.segment_index_t(r2);
            ret.emplace(std::make_tuple(seg1, t1, r1, seg2, t2, r2),
                        ruler.lineSliceAlong(r1, r2));
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

// Only returns labels (6 numbers), skips lineSliceAlong (no coordinate
// extraction)
inline PolylineChunkLabels
polyline_chunks_in_polygon(const RowVectors &polyline, //
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
        return {PolylineChunkLabel{0, 0.0, 0.0, ruler.N() - 2, 1.0,
                                   ruler.length()}};
    }
    const int N = intersections.size() + 2;
    Eigen::VectorXd ranges(N);
    {
        int idx = -1;
        ranges[++idx] = 0.0;
        for (auto &inter : intersections) {
            int seg_idx = std::get<2>(inter)[1];
            double t = std::get<1>(inter)[0];
            double r = ruler.range(seg_idx, t);
            ranges[++idx] = r;
        }
        ranges[++idx] = ruler.length();
    }
    RowVectorsNx2 midpoints(N - 1, 2);
    for (int i = 0; i < N - 1; ++i) {
        double rr = (ranges[i] + ranges[i + 1]) / 2.0;
        midpoints.row(i) = ruler.along(rr).head(2);
    }
    auto mask = point_in_polygon(midpoints, polygon);
    PolylineChunkLabels ret;
    for (int i = 0; i < N - 1; ++i) {
        double r1 = ranges[i];
        double r2 = ranges[i + 1];
        if (r2 <= r1 || mask[i] == 0) {
            continue;
        }
        auto [seg1, t1] = ruler.segment_index_t(r1);
        auto [seg2, t2] = ruler.segment_index_t(r2);
        ret.emplace_back(seg1, t1, r1, seg2, t2, r2);
    }
    return ret;
}

inline PolylineChunkLabels
polyline_chunks_in_polygon(const RowVectors &polyline, //
                           const Eigen::Ref<const RowVectorsNx2> &polygon,
                           bool is_wgs84 = false)
{
    auto fc = FastCrossing(is_wgs84);
    fc.add_polyline(polygon);
    fc.finish();
    return polyline_chunks_in_polygon(polyline, polygon, fc);
}

// Batch crop: for all polylines in `polylines_fc`, find chunks inside
// `polygon`. Returns map: polyline_index -> PolylineChunks (with coordinates)
inline std::map<int, PolylineChunks>
crop(const FastCrossing &polylines_fc,
     const Eigen::Ref<const RowVectorsNx2> &polygon,
     const FastCrossing &polygon_fc)
{
    // Use bbox of polygon to pre-filter candidate polylines
    Eigen::Vector2d pt0 = polygon.colwise().minCoeff();
    Eigen::Vector2d pt1 = polygon.colwise().maxCoeff();
    auto hits = polylines_fc.within(pt0, pt1, /*segment_wise=*/true,
                                    /*sort=*/false);
    // collect unique polyline indices
    std::set<int> poly_indices;
    for (auto &idx : hits) {
        poly_indices.insert(idx[0]);
    }
    std::map<int, PolylineChunks> ret;
    for (int pid : poly_indices) {
        const PolylineRuler *ruler = polylines_fc.polyline_ruler(pid);
        if (!ruler) {
            continue;
        }
        auto chunks =
            polyline_in_polygon(ruler->polyline(), polygon, polygon_fc);
        if (!chunks.empty()) {
            ret[pid] = std::move(chunks);
        }
    }
    return ret;
}

inline std::map<int, PolylineChunks>
crop(const FastCrossing &polylines_fc,
     const Eigen::Ref<const RowVectorsNx2> &polygon, bool is_wgs84 = false)
{
    auto polygon_fc = FastCrossing(is_wgs84);
    polygon_fc.add_polyline(polygon);
    polygon_fc.finish();
    return crop(polylines_fc, polygon, polygon_fc);
}

// Batch crop (labels only): same as crop but skips coordinate extraction
inline std::map<int, PolylineChunkLabels>
crop_labels(const FastCrossing &polylines_fc,
            const Eigen::Ref<const RowVectorsNx2> &polygon,
            const FastCrossing &polygon_fc)
{
    Eigen::Vector2d pt0 = polygon.colwise().minCoeff();
    Eigen::Vector2d pt1 = polygon.colwise().maxCoeff();
    auto hits = polylines_fc.within(pt0, pt1, /*segment_wise=*/true,
                                    /*sort=*/false);
    std::set<int> poly_indices;
    for (auto &idx : hits) {
        poly_indices.insert(idx[0]);
    }
    std::map<int, PolylineChunkLabels> ret;
    for (int pid : poly_indices) {
        const PolylineRuler *ruler = polylines_fc.polyline_ruler(pid);
        if (!ruler) {
            continue;
        }
        auto labels =
            polyline_chunks_in_polygon(ruler->polyline(), polygon, polygon_fc);
        if (!labels.empty()) {
            ret[pid] = std::move(labels);
        }
    }
    return ret;
}

inline std::map<int, PolylineChunkLabels>
crop_labels(const FastCrossing &polylines_fc,
            const Eigen::Ref<const RowVectorsNx2> &polygon,
            bool is_wgs84 = false)
{
    auto polygon_fc = FastCrossing(is_wgs84);
    polygon_fc.add_polyline(polygon);
    polygon_fc.finish();
    return crop_labels(polylines_fc, polygon, polygon_fc);
}

} // namespace cubao

#endif
