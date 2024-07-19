#include <string>

namespace hook {

/**
 * @brief Converts a size in bytes to a human-readable string with appropriate
 * units.
 *
 * This function takes a size in bytes and converts it to a more readable format
 * with units such as KB, MB, GB, etc. The result is a string that is easier to
 * understand for humans, providing the size in the largest possible unit
 * without losing precision.
 *
 * @param bytes The size in bytes to be converted.
 * @return A string representing the size in a human-readable format with
 * appropriate units.
 */
std::string prettyFormatSize(size_t bytes);

}  // namespace hook
