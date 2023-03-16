#ifndef CUBAO_POLYLINE_IN_POLYGON
#define CUBAO_POLYLINE_IN_POLYGON

#include "fast_crossing.hpp"

namespace cubao
{
    using PolylineChunks = std::map<std::tuple<int, // segment_index,
     double, // t
    double// range
    >, RowVectors>;
    polyline_in_polygon(const RowVectors &polyline, //
    const Eigen::Ref<const RowVectorsNx2> &polygon, bool is_wgs84 = false) {
        return {};
    }

} // namespace cubao

#endif
