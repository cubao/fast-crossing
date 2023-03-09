#ifndef CUBAO_QUIVER_HPP
#define CUBAO_QUIVER_HPP

#include <optional>

namespace cubao
{
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
// https://en.cppreference.com/w/cpp/types/climits
constexpr double NormEps = 1e-12; // DBL_EPSILON = 2.22045e-16
struct Arrow
{
    Arrow() {}
    Arrow(const Eigen::Vector3d &position) : position_(position) {}
    Arrow(const Eigen::Vector3d &position, const Eigen::Vector3d &direction)
        : Arrow(position)
    {
        this->direction(direction);
    }
    Eigen::Vector2i label() const { return {polyline_index_, segment_index_}; }
    Arrow &label(const Eigen::Vector2i &value)
    {
        polyline_index_ = value[0];
        segment_index_ = value[1];
        return *this;
    }
    Arrow &label(int polyline_index, int segment_index,
                 std::optional<double> t = std::nullopt,
                 std::optional<double> range = std::nullopt)
    {
        polyline_index_ = polyline_index;
        segment_index_ = segment_index;
        if (t) {
            t_ = *t;
        }
        if (range) {
            range_ = *range;
        }
        return *this;
    }

    int polyline_index() const { return polyline_index_; }
    Arrow &polyline_index(int value)
    {
        polyline_index_ = value;
        return *this;
    }
    int segment_index() const { return segment_index_; }
    Arrow &segment_index(int value)
    {
        segment_index_ = value;
        return *this;
    }

    double t() const { return t_; }
    Arrow &t(double value)
    {
        t_ = value;
        return *this;
    }
    double range() const { return range_; }
    Arrow &range(double value)
    {
        range_ = value;
        return *this;
    }
    void reset_index()
    {
        polyline_index_ = -1;
        segment_index_ = -1;
        t_ = NaN;
        range_ = NaN;
    }
    bool has_index(bool check_range = true) const
    {
        return polyline_index_ >= 0 && //
               segment_index_ >= 0 &&  //
               !std::isnan(t_) &&      //
               (!check_range || !std::isnan(range_));
    }

    const Eigen::Vector3d &position() const { return position_; }
    Arrow &position(const Eigen::Vector3d &position)
    {
        position_ = position;
        return *this;
    }
    const Eigen::Vector3d &direction() const { return direction_; }
    Arrow &direction(const Eigen::Vector3d &direction)
    {
        direction_ = direction;
        leftward_ = _unit_vector({-direction_[1], direction_[0], 0.0}, false);
        upward_ = direction_.cross(leftward_);
        return *this;
    }
    const Eigen::Vector3d leftward() const { return leftward_; }
    const Eigen::Vector3d &upward() const { return upward_; }
    Eigen::Matrix3d Frenet() const
    {
        Eigen::Matrix3d T;
        T.col(0) = direction_;
        T.col(1) = leftward_;
        T.col(2) = upward_;
        return T;
    }

    static double _heading(double east, double north)
    {
        // https://stackoverflow.com/questions/47909048/what-will-be-atan2-output-for-both-x-and-y-as-0
        static constexpr double DEG = 180.0 / 3.14159265358979323846;
        double h = std::atan2(east, north) * DEG;
        if (h < 0) {
            h += 360.0;
        }
        return h;
    }

    double heading() const { return _heading(direction_[0], direction_[1]); }
    static Eigen::Vector3d _heading(double value)
    {
        static constexpr double RAD = 3.14159265358979323846 / 180.0;
        value = (90.0 - value) * RAD;
        return Eigen::Vector3d(std::cos(value), std::sin(value), 0.0);
    }
    Arrow &heading(double value) { return direction(_heading(value)); }

    static Eigen::Vector3d _unit_vector(const Eigen::Vector3d &vector,
                                        bool with_eps = true)
    {
        Eigen::Vector3d d = vector;
        d /= (d.norm() + (with_eps ? NormEps : 0.0));
        return d;
    }

    // directly expose some non-invariant values on c++ side
    int polyline_index_ = -1;
    int segment_index_ = -1;
    double t_ = NaN;
    double range_ = NaN;
    Eigen::Vector3d position_{0.0, 0.0, 0.0}; // init to origin
  private:
    Eigen::Vector3d direction_{1.0, 0.0, 0.0}; // init to east
    Eigen::Vector3d leftward_{0.0, 1.0, 0.0};  // init to north
    Eigen::Vector3d upward_{0.0, 0.0, 1.0};    // init to up
};

struct Quiver
{
    const Eigen::Vector3d anchor_{0.0, 0.0, 0.0};
    const Eigen::Vector3d k_{1.0, 1.0, 1.0};
    const Eigen::Vector3d inv_k_{1.0, 1.0, 1.0};
    const bool is_wgs84_{false};
    Quiver() {}
    Quiver(const Eigen::Vector3d &anchor_lla)
        : anchor_(anchor_lla), k_(k(anchor_lla[1])), inv_k_(1.0 / k_.array()),
          is_wgs84_(true)
    {
    }

    // based on
    // https://github.com/cubao/headers/blob/main/include/cubao/crs_transform.hpp
    // https://github.com/mapbox/cheap-ruler-cpp
    static Eigen::Vector3d k(double latitude)
    {
        static constexpr double PI = 3.14159265358979323846;
        static constexpr double RE = 6378.137;
        static constexpr double FE = 1.0 / 298.257223563;
        static constexpr double E2 = FE * (2 - FE);
        static constexpr double RAD = PI / 180.0;
        static constexpr double MUL = RAD * RE * 1000.;
        double coslat = std::cos(latitude * RAD);
        double w2 = 1.0 / (1.0 - E2 * (1.0 - coslat * coslat));
        double w = std::sqrt(w2);
        return {MUL * w * coslat, MUL * w * w2 * (1.0 - E2), 1.0};
    }

    Arrow forwards(const Arrow &cur, double delta) const
    {
        auto copy = cur;
        copy.position_.array() +=
            delta * inv_k_.array() * copy.direction().array();
        return copy;
    }
    Arrow leftwards(const Arrow &cur, double delta) const
    {
        auto copy = cur;
        copy.position_.array() +=
            delta * inv_k_.array() * copy.leftward().array();
        return copy;
    }
    Arrow upwards(const Arrow &cur, double delta) const
    {
        auto copy = cur;
        copy.position_[2] += delta; // * k_[2];
        return copy;
    }
    Arrow towards(const Arrow &cur, const Eigen::Vector3d &delta,
                  bool update_direction = true) const
    {
        if (!delta.squaredNorm()) {
            return cur;
        }
        // update position (delta in Frenet)
        Eigen::Vector3d offset = delta[0] * cur.direction() +
                                 delta[1] * cur.leftward() +
                                 delta[2] * cur.upward();
        double norm = offset.norm();
        if (!norm) {
            return cur;
        }
        auto copy = cur;
        copy.position_.array() += inv_k_.array() * offset.array();
        if (update_direction) {
            copy.direction(offset / norm);
        }
        return copy;
    }

    Arrow update(const Arrow &cur, const Eigen::Vector3d &delta,
                 bool update_direction = true) const
    {
        double norm = delta.norm();
        if (!norm) {
            return cur;
        }
        auto copy = cur;
        // update position (delta in XYZ (like ENU))
        copy.position_.array() += inv_k_.array() * delta.array();
        if (update_direction) {
            copy.direction(delta / norm);
        }
        return copy;
    }
    // handles center,
    // searches

    // query arrows
    // filter arrows
    // aggregate arrows
    // label: polyline_index, seg_index, float
    // position, direction
};
} // namespace cubao
#endif
