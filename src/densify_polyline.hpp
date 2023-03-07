#ifndef CUBAO_DENSIFY_POLYLINE_HPP
#define CUBAO_DENSIFY_POLYLINE_HPP

#include <Eigen/Core>

namespace cubao
{
inline RowVectors densify_polyline(const Eigen::Ref<const RowVectors> &coords,
                                   double max_gap)
{
    if (coords.rows() < 2 || max_gap <= 0) {
        return coords;
    }
    std::vector<Eigen::Vector3d> xyzs;
    const int N = coords.rows();
    xyzs.reserve(N);
    for (int i = 0; i < N - 1; i++) {
        const Eigen::RowVector3d &curr = coords.row(i);
        const Eigen::RowVector3d &next = coords.row(i + 1);
        Eigen::RowVector3d dir = next - curr;
        double gap = dir.norm();
        if (gap == 0) {
            continue;
        }
        dir /= gap;
        if (gap <= max_gap) {
            xyzs.push_back(curr);
            continue;
        }
        RowVectors pos(1 + int(std::ceil(gap / max_gap)), 3);
        pos.col(0) = Eigen::ArrayXd::LinSpaced(pos.rows(), curr[0], next[0]);
        pos.col(1) = Eigen::ArrayXd::LinSpaced(pos.rows(), curr[1], next[1]);
        pos.col(2) = Eigen::ArrayXd::LinSpaced(pos.rows(), curr[2], next[2]);
        int M = pos.rows() - 1;
        int N = xyzs.size();
        xyzs.resize(N + M);
        Eigen::Map<RowVectors>(&xyzs[N][0], M, 3) = pos.topRows(M);
    }
    xyzs.push_back(coords.row(N - 1));
    return Eigen::Map<const RowVectors>(&xyzs[0][0], xyzs.size(), 3);
}
} // namespace cubao

#endif
