#ifndef CUBAO_NANO_KDTREE_HPP
#define CUBAO_NANO_KDTREE_HPP

#include <nanoflann.hpp>
#include <Eigen/Core>

namespace cubao {
using RowVectors = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowVectorsNx3 = RowVectors;
using RowVectorsNx2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;

// https://github.com/jlblancoc/nanoflann/blob/master/examples/utils.h
struct PointCloud
{
    struct Point
    {
        double x, y, z;
    };

    using coord_t = double;  //!< The type of each coordinate

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
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};


// https://github.com/jlblancoc/nanoflann/blob/master/examples/pointcloud_example.cpp
using KdTreeIndex = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 3 /* dim */>;

struct KdTree: PointCloud {
    KdTree() {}
    KdTree(const RowVectors &xyzs) {
        add(xyzs);
    }
    KdTree(const Eigen::Ref<const RowVectorsNx2> &xys) {
        add(xys);
    }

    Eigen::Map<const RowVectors> points(int i, int N) const {
        const double *data = &pts[i].x;
        return Eigen::Map<RowVectors>(data, N, 3);
    }
    Eigen::Map<const RowVectors> points() const {
        return points(0, pts.size());
    }

    void add(const RowVectors &xyzs) {
        int N = pts.size();
        int M = xyzs.rows();
        pts.resize(N + M);
        points(N, M) = xyzs;
    }
    void add(const Eigen::Ref<const RowVectorsNx2> &xys) {
        int N = pts.size();
        int M = xys.rows();
        pts.resize(N + M);
        auto points = this->points(N, M);
        points.leftCols(2) = xys;
        points.col(2).setZero();
    }

    // 
    private:
    Eigen::Map<RowVectors> points(int i, int N) {
        double *data = &pts[i].x;
        return Eigen::Map<RowVectors>(data, N, 3);
    }
    Eigen::Map<RowVectors> points() {
        return points(0, pts.size());
    }

    KdTreeIndex &index() const {
        if (!index_) {
            index_ = std::make_unique<KdTreeIndex>(3 /*dim*/, (const PointCloud&)*this, {10 /* max leaf */});
        }
        return *index_;
    }
    mutable std::unique_ptr<KdTreeIndex> index_;
};
}

#endif