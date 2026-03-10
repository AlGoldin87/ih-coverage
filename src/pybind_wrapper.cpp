#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "coverage_check.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace py = pybind11;

// Вспомогательная функция для преобразования vector<vector<float>> из numpy
std::vector<std::vector<float>> numpy_to_vector(py::array_t<float> input) {
    auto buf = input.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = ptr[i * cols + j];
        }
    }
    return result;
}

// Функция для одного столбца - рекомендуемая резкость
float recommend_sharpness(const std::vector<float>& data, int min_per_interval = 5) {
    if (data.empty()) return 1.0f;
    
    // Начинаем с крупных интервалов
    for (int n_int = 2; n_int <= 20; n_int++) {
        float sharpness = 2.0f / n_int;
        
        // Дискретизация
        float min_val = *std::min_element(data.begin(), data.end());
        float max_val = *std::max_element(data.begin(), data.end());
        float step = (max_val - min_val) / n_int;
        if (step < 1e-10f) step = 1.0f;
        
        std::vector<int> binned(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            int idx = static_cast<int>((data[i] - min_val) / step);
            idx = std::max(0, std::min(idx, n_int - 1));
            binned[i] = idx;
        }
        
        // Проверяем наполненность (БЕЗ C++17)
        std::unordered_map<int, int> counts;
        for (size_t i = 0; i < data.size(); i++) {
            counts[binned[i]]++;
        }
        
        bool all_ok = true;
        for (std::unordered_map<int, int>::iterator it = counts.begin(); it != counts.end(); ++it) {
            if (it->second < min_per_interval) {
                all_ok = false;
                break;
            }
        }
        
        if (all_ok) return sharpness;
    }
    
    return 2.0f / 20;  // если ничего не подошло
}

// Обёртка для Python - один столбец
float suggest_sharpness_1d(py::array_t<float> data, int min_per_interval = 5) {
    auto buf = data.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;
    
    std::vector<float> vec(ptr, ptr + size);
    return recommend_sharpness(vec, min_per_interval);
}

// Обёртка для Python
py::dict check_coverage_py(py::array_t<float> data,
                           float sharpness,
                           int min_per_interval = 5,
                           py::list feature_indices = py::list()) {

    auto cpp_data = numpy_to_vector(data);

    std::vector<int> indices;
    for (auto item : feature_indices) {
        indices.push_back(item.cast<int>());
    }

    auto report = check_data_coverage(cpp_data, sharpness, min_per_interval, indices);

    // Преобразуем отчёт в Python-словарь
    py::dict result;
    result["n_samples"] = report.n_samples;
    result["min_required"] = report.min_required;

    py::list features;
    float max_recommended = 0.0f;

    for (size_t i = 0; i < report.features.size(); i++) {
        const auto& feat = report.features[i];
        py::dict fdict;
        fdict["feature"] = feat.feature_idx;
        fdict["sharpness"] = feat.sharpness;
        fdict["n_intervals"] = feat.n_intervals;
        fdict["ok_count"] = feat.ok_count;
        fdict["warning_count"] = feat.warning_count;
        fdict["recommended_sharpness"] = feat.recommended_sharpness;

        if (feat.warning_count > 0) {
            max_recommended = std::max(max_recommended, feat.recommended_sharpness);
        }
        
        py::list intervals;
        for (size_t j = 0; j < feat.intervals.size(); j++) {
            const auto& iv = feat.intervals[j];
            py::dict idict;
            idict["interval"] = iv.interval;
            idict["count"] = iv.count;
            idict["ok"] = iv.ok;
            intervals.append(idict);
        }
        fdict["intervals"] = intervals;

        features.append(fdict);
    }

    result["features"] = features;
    result["recommended_sharpness"] = max_recommended;

    return result;
}

// Модуль
PYBIND11_MODULE(ih_coverage, m) {
    m.doc() = "Coverage check and sharpness optimization for IH library";

    m.def("check_coverage", &check_coverage_py,
          py::arg("data"),
          py::arg("sharpness"),
          py::arg("min_per_interval") = 5,
          py::arg("feature_indices") = py::list(),
          "Check coverage of intervals for given features\n\n"
          "Args:\n"
          "    data: 2D numpy array (rows x cols)\n"
          "    sharpness: current sharpness value\n"
          "    min_per_interval: minimum observations per interval (default 5)\n"
          "    feature_indices: list of feature indices to check (default all)\n\n"
          "Returns:\n"
          "    dict with coverage report and recommendations");

    m.def("discretize", [](py::array_t<float> data, float sharpness) {
        auto buf = data.request();
        float* ptr = static_cast<float*>(buf.ptr);
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];
        
        // Результат: сначала дискретизируем по столбцам
        std::vector<std::vector<int>> col_results(cols, std::vector<int>(rows));
        
        // 1. Дискретизация по столбцам
        for (size_t j = 0; j < cols; j++) {
            // Собираем столбец j
            std::vector<float> column(rows);
            for (size_t i = 0; i < rows; i++) {
                column[i] = ptr[i * cols + j];
            }
            
            // Дискретизируем столбец
            auto binned_col = discretize_feature(column, sharpness);
            
            // Записываем в колоночный результат
            for (size_t i = 0; i < rows; i++) {
                col_results[j][i] = binned_col[i];
            }
        }
        
        // 2. Транспонируем обратно в строки (rows × cols)
        std::vector<std::vector<int>> final_result(rows, std::vector<int>(cols));
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                final_result[i][j] = col_results[j][i];
            }
        }
        
        return final_result;
    }, py::arg("data"), py::arg("sharpness"),
       "Discretize data using given sharpness");

    m.def("suggest_sharpness", &suggest_sharpness_1d,
          py::arg("data"),
          py::arg("min_per_interval") = 5,
          "Suggest optimal sharpness for a single column");
}
