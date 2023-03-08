#ifndef CUBAO_NANO_KDTREE_HPP
#define CUBAO_NANO_KDTREE_HPP

#include <nanoflann.hpp>
#include <Eigen/Core>
#include <memory>
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
    nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 3 /* dim */>;

struct KdTree : PointCloud
{
    KdTree() {}
    KdTree(const RowVectors &xyzs) { add(xyzs); }
    KdTree(const Eigen::Ref<const RowVectorsNx2> &xys) { add(xys); }

    Eigen::Map<const RowVectors> points(int i, int N) const
    {
        const double *data = &pts[i][0];
        return Eigen::Map<const RowVectors>(data, N, 3);
    }
    Eigen::Map<const RowVectors> points() const
    {
        return points(0, pts.size());
    }

    void add(const RowVectors &xyzs)
    {
        int N = pts.size();
        int M = xyzs.rows();
        pts.resize(N + M);
        points(N, M) = xyzs;
        index_.reset();
    }
    void add(const Eigen::Ref<const RowVectorsNx2> &xys)
    {
        int N = pts.size();
        int M = xys.rows();
        pts.resize(N + M);
        auto points = this->points(N, M);
        points.leftCols(2) = xys;
        points.col(2).setZero();
        index_.reset();
    }
    void clear()
    {
        pts.clear();
        index_.reset();
    }

    std::pair<int, double> nearest(const Eigen::Vector3d &position, //
                                   bool return_squared_l2 = false) const
    {
        size_t ret_index;
        double out_dist_sqr;
        nanoflann::KNNResultSet<double> resultSet(1);
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
        auto ret = nearest(pts[index], 2, return_squared_l2);
        return ret.first[0] == index
                   ? std::make_pair(ret.first[1], ret.second[1])
                   : std::make_pair(ret.first[0], ret.second[0]);
    }

    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    nearest(const Eigen::Vector3d &position, //
            int k,                           //
            bool sorted = true,              //
            bool return_squared_l2 = false) const
    {
        if (k <= 0) {
            return std::make_pair(Eigen::VectorXi(0), Eigen::VectorXd(0));
        }
        std::vector<size_t> ret_index(k);
        std::vector<double> out_dist_sqr(k);
        nanoflann::KNNResultSet<double> resultSet(k);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        auto params = nanoflann::SearchParams();
        params.sorted = sorted;
        index().findNeighbors(resultSet, position.data(), params);
        std::vector<int> indexes;
        std::vector<double> distances;
        for (size_t i = 0; i < resultSet.size(); i++) {
            indexes.push_back(ret_index[i]);
            distances.push_back(out_dist_sqr[i]);
        }
        return std::make_pair(
            Eigen::VectorXi::Map(&indexes[0], indexes.size()),
            return_squared_l2
                ? Eigen::VectorXd::Map(&distances[0], distances.size())
                : Eigen::VectorXd::Map(&distances[0], distances.size())
                      .cwiseSqrt()
                      .eval());
    }
    std::pair<Eigen::VectorXi, Eigen::VectorXd>
    nearest(const Eigen::Vector3d &position, //
            double radius,                   //
            bool sorted = true,              //
            bool return_squared_l2 = false) const
    {
        auto params = nanoflann::SearchParams();
        params.sorted = sorted;
        std::vector<std::pair<size_t, double>> indices_dists;
        index().radiusSearch(position.data(), //
                             radius * radius, indices_dists, params);

        std::vector<int> indexes;
        std::vector<double> distances;
        for (size_t i = 0; i < indices_dists.size(); i++) {
            indexes.push_back(indices_dists[i].first);
            distances.push_back(indices_dists[i].second);
        }
        return std::make_pair(
            Eigen::VectorXi::Map(&indexes[0], indexes.size()),
            return_squared_l2
                ? Eigen::VectorXd::Map(&distances[0], distances.size())
                : Eigen::VectorXd::Map(&distances[0], distances.size())
                      .cwiseSqrt()
                      .eval());
    }

    //
  private:
    Eigen::Map<RowVectors> points(int i, int N)
    {
        double *data = &pts[i][0];
        return Eigen::Map<RowVectors>(data, N, 3);
    }
    Eigen::Map<RowVectors> points() { return points(0, pts.size()); }

    KdTreeIndex &index() const
    {
        if (!index_) {
            index_ = std::make_unique<KdTreeIndex>(
                3 /*dim*/, (const PointCloud &)*this,
                nanoflann::KDTreeSingleIndexAdaptorParams(10));
        }
        return *index_;
    }
    mutable std::unique_ptr<KdTreeIndex> index_; // should be noncopyable
};
} // namespace cubao

#endif
