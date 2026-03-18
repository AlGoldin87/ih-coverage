import numpy as np
from ih_coverage import suggest_sharpness

# Наши данные X1
x1 = np.array([
    79.49, 86.57, 86.35, 87.42, 93.39, 90.56, 91.95, 96.93, 97.80, 97.79,
    97.60, 98.09, 97.76, 95.39, 95.62, 95.20, 95.08, 92.58, 91.02, 89.75,
    90.00, 88.68, 86.61, 86.00, 84.26, 81.12, 79.18, 78.08, 77.23, 74.83,
    72.40, 71.41, 70.02, 67.07, 64.42, 62.31, 62.19, 59.41, 55.30, 54.90,
    54.29, 51.06, 48.18, 49.89, 48.26, 49.46, 50.19, 51.77, 52.88, 53.44
], dtype=np.float32)

print("="*60)
print("ТЕСТИРОВАНИЕ suggest_sharpness С ICC")
print("="*60)

# Проверим определение минимального шага
print(f"\nМинимальный шаг по данным: вычисляется автоматически")

# Тестируем с разными alpha
print("\nРезультаты при разных alpha:\n")

for alpha in [0.5, 1.0, 2.0, 5.0]:
    s = suggest_sharpness(x1, min_per_interval=5, alpha=alpha)
    intervals = int(round(2.0 / s))
    print(f"alpha = {alpha:3.1f} → резкость = {s:.2f}  (интервалов: {intervals})")

# Дополнительно: посмотрим, что будет, если изменить min_per_interval
print("\n" + "="*60)
print("ВЛИЯНИЕ min_per_interval (при alpha=1.0)")
print("="*60)

for min_cnt in [3, 5, 10, 20]:
    s = suggest_sharpness(x1, min_per_interval=min_cnt, alpha=1.0)
    intervals = int(round(2.0 / s))
    print(f"min_per_interval = {min_cnt:2d} → резкость = {s:.2f} (интервалов: {intervals})")