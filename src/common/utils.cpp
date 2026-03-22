#include "common/utils.hpp"
#include "common/constants.hpp"
#include "core/coreset.hpp"

namespace kmeans {
	// Create a 5D feature vector from BGR color and normalized spatial coordinates, scaled by the color_scale and
	// spatial_scale arguments
	// We scale so that they are in roughly the same range and to not let color or space dominate the distance metric
	cv::Vec<float, 5> makeFeature(
		const cv::Vec3f& bgr,
		float x01,
		float y01)
	{
		return cv::Vec<float, 5>(
			bgr[0] * COLOR_SCALE,
			bgr[1] * COLOR_SCALE,
			bgr[2] * COLOR_SCALE,
			x01 * SPATIAL_SCALE,
			y01 * SPATIAL_SCALE
		);
	}
}