#ifndef CUBAO_POINT_IN_POLYGON_HPP
#define CUBAO_POINT_IN_POLYGON_HPP

#include <Eigen/Core>

namespace cubao
{
using RowVectorsNx2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
namespace agg
{
enum path_commands
{
    path_cmd_stop = 0,        //----path_cmd_stop
    path_cmd_move_to = 1,     //----path_cmd_move_to
    path_cmd_line_to = 2,     //----path_cmd_line_to
    path_cmd_curve3 = 3,      //----path_cmd_curve3
    path_cmd_curve4 = 4,      //----path_cmd_curve4
    path_cmd_curveN = 5,      //----path_cmd_curveN
    path_cmd_catrom = 6,      //----path_cmd_catrom
    path_cmd_ubspline = 7,    //----path_cmd_ubspline
    path_cmd_end_poly = 0x0F, //----path_cmd_end_poly
    path_cmd_mask = 0x0F      //----path_cmd_mask
};
}

// https://github.com/matplotlib/matplotlib/blob/7303d81eb2390208d3bac67dd51d390b585e411f/src/_path.h#L67-L233
// The following function was found in the Agg 2.3 examples
// (interactive_polygon.cpp). It has been generalized to work on (possibly
// curved) polylines, rather than just polygons.  The original comments have
// been kept intact.
//  -- Michael Droettboom 2007-10-02
//
//======= Crossings Multiply algorithm of InsideTest ========================
//
// By Eric Haines, 3D/Eye Inc, erich@eye.com
//
// This version is usually somewhat faster than the original published in
// Graphics Gems IV; by turning the division for testing the X axis crossing
// into a tricky multiplication test this part of the test became faster,
// which had the additional effect of making the test for "both to left or
// both to right" a bit slower for triangles than simply computing the
// intersection each time.  The main increase is in triangle testing speed,
// which was about 15% faster; all other polygon complexities were pretty much
// the same as before.  On machines where division is very expensive (not the
// case on the HP 9000 series on which I tested) this test should be much
// faster overall than the old code.  Your mileage may (in fact, will) vary,
// depending on the machine and the test data, but in general I believe this
// code is both shorter and faster.  This test was inspired by unpublished
// Graphics Gems submitted by Joseph Samosky and Mark Haigh-Hutchinson.
// Related work by Samosky is in:
//
// Samosky, Joseph, "SectionView: A system for interactively specifying and
// visualizing sections through three-dimensional medical image data",
// M.S. Thesis, Department of Electrical Engineering and Computer Science,
// Massachusetts Institute of Technology, 1993.
//
// Shoot a test ray along +X axis.  The strategy is to compare vertex Y values
// to the testing point's Y and quickly discard edges which are entirely to one
// side of the test ray.  Note that CONVEX and WINDING code can be added as
// for the CrossingsTest() code; it is left out here for clarity.
//
// Input 2D polygon _pgon_ with _numverts_ number of vertices and test point
// _point_, returns 1 if inside, 0 if outside.
inline Eigen::VectorXi
point_in_polygon(const Eigen::Ref<const RowVectorsNx2> &points,
                 const Eigen::Ref<const RowVectorsNx2> &polygon // or polyline
)
{
    if (!points.rows() || !polygon.rows()) {
        throw std::invalid_argument("invalid polygon, or empty points");
    }

    size_t n = points.rows();
    Eigen::VectorXi inside_flag(n);
    inside_flag.setZero();
    const int N = polygon.rows();

    uint8_t yflag1;
    double vtx0, vty0, vtx1, vty1;
    double tx, ty;
    double sx, sy;
    double x, y;
    bool all_done;

    std::vector<uint8_t> yflag0(n);
    std::vector<uint8_t> subpath_flag(n);
    unsigned path_id = 0;
    unsigned code = 0;
    unsigned vertex;

    do {
        if (code != agg::path_cmd_move_to) {
            if (path_id >= N) {
                x = 0.0;
                y = 0.0;
                vertex = agg::path_cmd_stop;
            } else {
                x = polygon(path_id, 0);
                y = polygon(path_id, 1);
                vertex = path_id == 0 ? agg::path_cmd_move_to
                                      : agg::path_cmd_line_to;
                ++path_id;
            }
            code = vertex;
            if (code == agg::path_cmd_stop ||
                (code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
                continue;
            }
        }

        sx = vtx0 = vtx1 = x;
        sy = vty0 = vty1 = y;

        for (size_t i = 0; i < n; ++i) {
            ty = points(i, 1);

            if (std::isfinite(ty)) {
                // get test bit for above/below X axis
                yflag0[i] = (vty0 >= ty);

                subpath_flag[i] = 0;
            }
        }

        do {
            if (path_id >= N) {
                x = 0.0;
                y = 0.0;
                vertex = agg::path_cmd_stop;
            } else {
                x = polygon(path_id, 0);
                y = polygon(path_id, 1);
                vertex = path_id == 0 ? agg::path_cmd_move_to
                                      : agg::path_cmd_line_to;
                ++path_id;
            }
            code = vertex;

            // The following cases denote the beginning on a new subpath
            if (code == agg::path_cmd_stop ||
                (code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
                x = sx;
                y = sy;
            } else if (code == agg::path_cmd_move_to) {
                break;
            }

            for (size_t i = 0; i < n; ++i) {
                tx = points(i, 0);
                ty = points(i, 1);

                if (!(std::isfinite(tx) && std::isfinite(ty))) {
                    continue;
                }

                yflag1 = (vty1 >= ty);
                // Check if endpoints straddle (are on opposite sides) of
                // X axis (i.e. the Y's differ); if so, +X ray could
                // intersect this edge.  The old test also checked whether
                // the endpoints are both to the right or to the left of
                // the test point.  However, given the faster intersection
                // point computation used below, this test was found to be
                // a break-even proposition for most polygons and a loser
                // for triangles (where 50% or more of the edges which
                // survive this test will cross quadrants and so have to
                // have the X intersection computed anyway).  I credit
                // Joseph Samosky with inspiring me to try dropping the
                // "both left or both right" part of my code.
                if (yflag0[i] != yflag1) {
                    // Check intersection of pgon segment with +X ray.
                    // Note if >= point's X; if so, the ray hits it.  The
                    // division operation is avoided for the ">=" test by
                    // checking the sign of the first vertex wrto the test
                    // point; idea inspired by Joseph Samosky's and Mark
                    // Haigh-Hutchinson's different polygon inclusion
                    // tests.
                    if (((vty1 - ty) * (vtx0 - vtx1) >=
                         (vtx1 - tx) * (vty0 - vty1)) == yflag1) {
                        subpath_flag[i] ^= 1;
                    }
                }

                // Move to the next pair of vertices, retaining info as
                // possible.
                yflag0[i] = yflag1;
            }

            vtx0 = vtx1;
            vty0 = vty1;

            vtx1 = x;
            vty1 = y;
        } while (code != agg::path_cmd_stop &&
                 (code & agg::path_cmd_end_poly) != agg::path_cmd_end_poly);

        all_done = true;
        for (size_t i = 0; i < n; ++i) {
            tx = points(i, 0);
            ty = points(i, 1);

            if (!(std::isfinite(tx) && std::isfinite(ty))) {
                continue;
            }

            yflag1 = (vty1 >= ty);
            if (yflag0[i] != yflag1) {
                if (((vty1 - ty) * (vtx0 - vtx1) >=
                     (vtx1 - tx) * (vty0 - vty1)) == yflag1) {
                    subpath_flag[i] = subpath_flag[i] ^ true;
                }
            }
            inside_flag[i] |= subpath_flag[i];
            if (inside_flag[i] == 0) {
                all_done = false;
            }
        }

        if (all_done) {
            break;
        }
    } while (code != agg::path_cmd_stop);
    return inside_flag;
}
} // namespace cubao

#endif
