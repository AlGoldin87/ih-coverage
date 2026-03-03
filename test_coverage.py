# test_coverage.py
import numpy as np
import ih_coverage

# Тестовые данные (маленькие для примера)
data = np.array([
    [1.2, 2.3, 3.4],
    [1.5, 2.1, 3.6],
    [1.8, 2.7, 3.9],
    [2.1, 3.0, 4.2],
    [2.4, 3.3, 4.5]
], dtype=np.float32)

print("=" * 60)
print("ТЕСТ МОДУЛЯ 1: ДИСКРЕТИЗАЦИЯ")
print("=" * 60)

# Модуль 1: просто дискретизация
sharpness = 0.5  # 2/0.5 = 4 интервала
binned = ih_coverage.discretize(data, sharpness)

print(f"Резкость: {sharpness} → {int(2/sharpness)} интервалов")
print("Исходные данные:")
print(data)
print("Дискретизированные данные:")
print(binned)
print()

print("=" * 60)
print("ТЕСТ МОДУЛЯ 2: ПРОВЕРКА ПОКРЫТИЯ")
print("=" * 60)

# Модуль 2: проверка покрытия
report = ih_coverage.check_coverage(
    data=data,
    sharpness=0.5,
    min_per_interval=2,  # минимум 2 наблюдения на интервал
    feature_indices=[0, 1]  # проверим только первые два признака
)

print(f"Всего строк: {report['n_samples']}")
print(f"Порог наполнения: {report['min_required']}")
print(f"Рекомендуемая резкость: {report['recommended_sharpness']:.2f}")
print()

for feat in report['features']:
    print(f"\n--- ПРИЗНАК X{feat['feature'] + 1} ---")
    print(f"  Интервалов: {feat['n_intervals']}")
    print(f"  OK: {feat['ok_count']}, Проблемных: {feat['warning_count']}")
    print(f"  Рекомендация: резкость = {feat['recommended_sharpness']:.2f}")
    
    for iv in feat['intervals']:
        status = "✅" if iv['ok'] else "⚠️ МАЛО"
        print(f"    Интервал {iv['interval']}: {iv['count']} obs {status}")