#ifndef CUBAO_QUIVER_HPP
#define CUBAO_QUIVER_HPP

namespace cubao
{
struct Arrow
{
    int polyline_index = -1;
    int segment_index = -1;
    double t = -1.0;
    double range = -1.0;
    Eigen::Vector3d position{0.0, 0.0, 0.0};  // origin
    Eigen::Vector3d direction{0.0, 0.0, 1.0}; // upwards
};

struct Quiver
{
    const Eigen::Vector3d anchor_{0.0, 0.0, 0.0};
    const Eigen::Vector3d k_{1.0, 1.0, 1.0};
    const bool is_wgs84_{false};
    Quiver(bool is_wgs84 = false) : is_wgs84_(is_wgs84) {}
    Quiver(const Eigen::Vector3d &anchor_lla)
        : anchor_(anchor_lla), k_(k(anchor_lla[1])), is_wgs84_(true)
    {
    }

    // based on
    // https://github.com/cubao/headers/blob/main/include/cubao/crs_transform.hpp
    // https://github.com/mapbox/cheap-ruler-cpp
    inline static Eigen::Vector3d k(double latitude)
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
