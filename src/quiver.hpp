#ifndef CUBAO_QUIVER_HPP
#define CUBAO_QUIVER_HPP

#include <optional>

namespace cubao
{
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
struct Arrow
{
    // POD as it can be, all public in c++ (should not expose to python side)
    int polyline_index_ = -1;
    int segment_index_ = -1;
    double t_ = NaN;
    double range_ = NaN;
    Eigen::Vector3d position_{0.0, 0.0, 0.0};  // init to origin
    Eigen::Vector3d direction_{0.0, 0.0, 1.0}; // init to upwards

    Arrow() {}
    Arrow(const Eigen::Vector3d &position,
          const Eigen::Vector3d &direction = {0.0, 0.0, 1.0})
        : position_(position), direction_(direction)
    {
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
               (check_range || !std::isnan(range_));
    }

    Eigen::Vector3d position() const { return position_; }
    Arrow &position(const Eigen::Vector3d &position)
    {
        position_ = position;
        return *this;
    }
    Eigen::Vector3d direction() const { return direction_; }
    Arrow &direction(const Eigen::Vector3d &direction,
                     bool need_normalize = false)
    {
        direction_ = direction;
        if (need_normalize) {
            direction_ /= (direction_.norm() + 1e-18);
        }
        return *this;
    }
    Eigen::Vector3d leftward() const
    {
        Eigen::Vector3d vec{-direction_[1], direction_[0], 0.0};
        if (direction_[2] == 0.0 || std::fabs(direction_[2]) < 1e-6) {
            return vec;
        }
        vec /= (vec.norm() + 1e-18);
        return vec;
    }

    double heading() const
    {
        static constexpr double DEG = 180.0 / 3.14159265358979323846;
        double h = std::atan2(direction_[0], direction_[1]) * DEG;
        if (h < 0) {
            h += 360.0;
        }
        return h;
    }
    Arrow &heading(double value)
    {
        static constexpr double RAD = 3.14159265358979323846 / 180.0;
        value = (90.0 - value) * RAD;
        direction_[0] = std::cos(value);
        direction_[1] = std::sin(value);
        direction_[2] = 0.0;
        return *this;
    }
};

struct Quiver
{
    const Eigen::Vector3d anchor_{0.0, 0.0, 0.0};
    const Eigen::Vector3d k_{1.0, 1.0, 1.0};
    const Eigen::Vector3d inv_k_{1.0, 1.0, 1.0};
    const bool is_wgs84_{false};
    Quiver() {}
    Quiver(const Eigen::Vector3d &anchor_lla)
        : anchor_(anchor_lla), k_(k(anchor_lla[1])), inv_k_(1.0 / k_.array()), is_wgs84_(true)
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

    Arrow forward(const Arrow &cur, double delta) const
    {
        auto copy = cur;
        copy.position_.array() += delta * inv_k_.array() * copy.direction().array();
        return copy;
    }
    Arrow leftward(const Arrow &cur, double delta) const
    {
        auto copy = cur;
        copy.position_.array() += delta * inv_k_.array() * copy.leftward().array();
        return copy;
    }
    Arrow upward(const Arrow &cur, double delta) const
    {
        auto copy = cur;
        copy.position_[2] += delta; // * k_[2];
        return copy;
    }
    Arrow update(const Arrow &cur, const Eigen::Vector3d &delta,
                 bool keep_direction = false) const
    {
        double norm = delta.norm();
        if (!norm) {
            return cur;
        }
        auto copy = cur;
        copy.position_.array() += inv_k_.array() * delta.array();
        if (!keep_direction) {
            copy.direction_ = delta / norm;
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
