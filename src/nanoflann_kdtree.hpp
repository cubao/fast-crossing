// should sync
// - https://github.com/cubao/fast-crossing/blob/master/src/nanoflann_kdtree.hpp
// -
// https://github.com/cubao/headers/tree/main/include/cubao/nanoflann_kdtree.hpp

#ifndef CUBAO_NANO_KDTREE_HPP
#define CUBAO_NANO_KDTREE_HPP

#include <nanoflann.hpp>
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace cubao
{
using RowVectors = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowVectorsNx3 = RowVectors;
using RowVectorsNx2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

// https://github.com/jlblancoc/nanoflann/blob/master/examples/utils.h
struct PointCloud
{
    using Point = Eigen::Vector3d;
    using coord_t = double; //!< The type of each coordinate
    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx][0];
        else if (dim == 1)
            return pts[idx][1];
        else
            return pts[idx][2];
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const
    {
        return false;
    }
};

// https://github.com/jlblancoc/nanoflann/blob/master/examples/pointcloud_example.cpp
using KdTreeIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud, double, int>, //
    PointCloud,                                                    //
    3 /* dim */,                                                   //
    int /* AccessorType */
    >;

struct KdTree : PointCloud
{
    KdTree(int leafsize = 10) : leafsize_(leafsize) {}
    KdTree(const RowVectors &xyzs) { add(xyzs); }
    KdTree(const Eigen::Ref<const RowVectorsNx2> &xys) { add(xys); }

    Eigen::Map<const RowVectors> points() const
    {
        return Eigen::Map<const RowVectors>(&pts[0][0], pts.size(), 3);
    }

    void add(const RowVectors &xyzs)
    {
        int N = pts.size();
        int M = xyzs.rows();
        pts.resize(N + M);
        points_(N, M) = xyzs;
        index_.reset();
    }
    void add(const Eigen::Ref<const RowVectorsNx2> &xys)
    {
        int N = pts.size();
        int M = xys.rows();
        pts.resize(N + M);
        auto points = this->points_(N, M);
        points.leftCols(2) = xys;
        points.col(2).setZero();
        index_.reset();
    }

    void reset()
    {
        pts.clear();
        index_.reset();
    }
    void reset_index() { index_.reset(); }
    void build_index(bool force = false) { index(force); }

    int leafsize() const { return leafsize_; }
    void set_leafsize(size_t value) { leafsize_ = value; }

    // https://github.com/kzampog/cilantro/tree/master
    std::pair<int, double> nearest(const Eigen::Vector3d &position, //
                                   bool return_squared_l2 = false) const
    {
        int ret_index;
        double out_dist_sqr;
        auto resultSet = nanoflann::KNNResultSet<double, int, int>(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        index().findNeighbors(resultSet, position.data(),
                              nanoflann::SearchParams());
        return std::make_pair(ret_index, //
                              return_squared_l2 ? out_dist_sqr
                                                : std::sqrt(out_dist_sqr));
    }
    std::pair<int, double> nearest(int index, //
                                   bool return_squared_l2 = false) const
    {
        auto ret = nearest(pts[index], 2, false, return_squared_l2);
        return ret.first[0] == index
                   ? std::make_pair(ret.first[1], ret.second[1])
                   : std::make_pair(ret.first[0], ret.second[0]);
    }

    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    nearest(const Eigen::Vector3d &position, //
            int k,                           //
            bool sort = true,                //
            bool return_squared_l2 = false) const
    {
        auto indexes = std::vector<int>(k);
        auto distances = std::vector<double>(k);
        auto resultSet = nanoflann::KNNResultSet<double, int, int>(k);
        resultSet.init(&indexes[0], &distances[0]);
        auto params = nanoflann::SearchParams();
        params.sorted = sort;
        index().findNeighbors(resultSet, position.data(), params);
        const int N = resultSet.size();
        return std::make_pair(
            Eigen::VectorXi::Map(&indexes[0], N),
            return_squared_l2
                ? Eigen::VectorXd::Map(&distances[0], N)
                : Eigen::VectorXd::Map(&distances[0], N).cwiseSqrt().eval());
    }
    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    nearest(const Eigen::Vector3d &position, //
            double radius,                   //
            bool sort = true,                //
            bool return_squared_l2 = false) const
    {
        auto params = nanoflann::SearchParams();
        params.sorted = sort;
        std::vector<std::pair<int, double>> indices_dists;
        index().radiusSearch(position.data(), //
                             radius * radius, //
                             indices_dists, params);

        const int N = indices_dists.size();
        Eigen::VectorXi indexes(N);
        Eigen::VectorXd distances(N);
        for (int i = 0; i < N; i++) {
            indexes[i] = indices_dists[i].first;
            distances[i] = indices_dists[i].second;
        }
        return std::make_pair(indexes, return_squared_l2
                                           ? distances
                                           : distances.cwiseSqrt().eval());
    }

    // Batch version of nearest(position, k) / nearest(position, radius).
    //
    // `queries` is (N, 3) (only the first 3 columns are used, so (N, 2) input
    // can be passed through `to_Nx3`).
    //
    // Exactly one of `k` / `radius` should be given:
    //   k       : width is min(k, #points), one pass, every query gets the
    //             same number of slots.
    //   radius  : width is `max_hits`, or - when `max_hits` is null - the
    //             widest hit count of the batch. Two passes (gather, then
    //             fill), because the width is only known after the first pass.
    //
    // Returns
    //   counts   : (N,)   real number of hits per query (before truncation)
    //   indexes  : (N, H) row `i` holds counts[i] valid point indexes, -1 pad
    //   distances: (N, H) matching distances, inf pad
    //
    // Results are always sorted by distance: a fixed-width output with
    // truncation is only well defined when the order is. The traversal state
    // (one KdTreeIndex, one scratch buffer) is reused across queries.
    std::tuple<Eigen::VectorXi, Eigen::MatrixXi, Eigen::MatrixXd>
    nearest_batch(const RowVectors &queries, std::optional<int> k,
                  std::optional<double> radius, std::optional<int> max_hits,
                  bool return_squared_l2 = false) const
    {
        if (k.has_value() == radius.has_value()) {
            throw std::invalid_argument(
                "specify exactly one of k / radius for nearest_batch");
        }
        if (k && *k <= 0) {
            throw std::invalid_argument("k should be positive");
        }
        if (radius && *radius < 0) {
            throw std::invalid_argument("radius should be non-negative");
        }
        if (max_hits && *max_hits < 0) {
            throw std::invalid_argument("max_hits should be non-negative");
        }

        const int N = queries.rows();
        const int M = static_cast<int>(pts.size());
        const double inf = std::numeric_limits<double>::infinity();

        Eigen::VectorXi counts = Eigen::VectorXi::Zero(N);
        if (N == 0 || M == 0) {
            const int H = max_hits ? *max_hits : (k ? std::min(*k, M) : 0);
            return {counts, Eigen::MatrixXi::Constant(N, H, -1),
                    Eigen::MatrixXd::Constant(N, H, inf)};
        }

        if (k) {
            const int H = std::min(*k, M);
            Eigen::MatrixXi indexes = Eigen::MatrixXi::Constant(N, H, -1);
            Eigen::MatrixXd distances = Eigen::MatrixXd::Constant(N, H, inf);
            if (H == 0) {
                return {counts, indexes, distances};
            }
            std::vector<int> scratch_indexes(H);
            std::vector<double> scratch_dists(H);
            auto params = nanoflann::SearchParams();
            params.sorted = true;
            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d q = queries.row(i);
                auto resultSet = nanoflann::KNNResultSet<double, int, int>(H);
                resultSet.init(&scratch_indexes[0], &scratch_dists[0]);
                index().findNeighbors(resultSet, q.data(), params);
                const int n = static_cast<int>(resultSet.size());
                counts[i] = n;
                for (int j = 0; j < n; ++j) {
                    indexes(i, j) = scratch_indexes[j];
                    distances(i, j) = return_squared_l2
                                          ? scratch_dists[j]
                                          : std::sqrt(scratch_dists[j]);
                }
            }
            return {counts, indexes, distances};
        }

        // radius mode: first pass gathers, second pass fills
        std::vector<std::vector<std::pair<int, double>>> found(N);
        int width = max_hits ? *max_hits : 0;
        auto params = nanoflann::SearchParams();
        params.sorted = true;
        const double r2 = (*radius) * (*radius);
        for (int i = 0; i < N; ++i) {
            Eigen::Vector3d q = queries.row(i);
            auto &hits = found[i];
            if (*radius <= 0.0) {
                continue;
            }
            index().radiusSearch(q.data(), r2, hits, params);
            if (!max_hits) {
                width = std::max(width, static_cast<int>(hits.size()));
            }
        }
        const int H = width;
        Eigen::MatrixXi indexes = Eigen::MatrixXi::Constant(N, H, -1);
        Eigen::MatrixXd distances = Eigen::MatrixXd::Constant(N, H, inf);
        for (int i = 0; i < N; ++i) {
            const auto &hits = found[i];
            counts[i] = static_cast<int>(hits.size());
            const int n = std::min(H, static_cast<int>(hits.size()));
            for (int j = 0; j < n; ++j) {
                indexes(i, j) = hits[j].first;
                distances(i, j) = return_squared_l2 ? hits[j].second
                                                    : std::sqrt(hits[j].second);
            }
        }
        return {counts, indexes, distances};
    }

    //
  private:
    Eigen::Map<RowVectors> points_(int i, int N)
    {
        double *data = &pts[i][0];
        return Eigen::Map<RowVectors>(data, N, 3);
    }
    Eigen::Map<RowVectors> points_() { return points_(0, pts.size()); }

    // mutable index
    //  CppCon 2017: Kate Gregory “10 Core Guidelines You Need to Start Using
    //  Now” https://youtu.be/XkDEzfpdcSg?t=1093
    //      Restoring const-correctness
    mutable std::unique_ptr<KdTreeIndex> index_; // should be noncopyable
    KdTreeIndex &index(bool force_rebuild = false) const
    {
        if (force_rebuild || !index_) {
            index_ = std::make_unique<KdTreeIndex>(
                3 /*dim*/, (const PointCloud &)*this,
                nanoflann::KDTreeSingleIndexAdaptorParams(leafsize_));
        }
        return *index_;
    }
    int leafsize_ = 10;
};
} // namespace cubao

#endif
