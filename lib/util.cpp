#include <iomanip>
#include <sstream>
#include <string>

namespace hook {
std::string prettyFormatSize(size_t bytes) {
    const char* sizes[] = {"B", "KB", "MB", "GB"};
    size_t order = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024 && order < sizeof(sizes) / sizeof(sizes[0]) - 1) {
        order++;
        size /= 1024;
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << size << " " << sizes[order];
    return out.str();
}

}  // namespace hook
