#!/bin/bash
# Скрипт для запуска MLflow UI на Linux/Mac

# Получаем директорию скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MLRUNS_DIR="$SCRIPT_DIR/mlruns"

# Проверяем существование директории mlruns
if [ ! -d "$MLRUNS_DIR" ]; then
    echo "⚠️  Директория mlruns не найдена. Создаю: $MLRUNS_DIR"
    mkdir -p "$MLRUNS_DIR"
fi

echo "============================================================"
echo "🚀 ЗАПУСК MLFLOW UI"
echo "============================================================"
echo "📁 Директория mlruns: $MLRUNS_DIR"
echo "🌐 MLflow UI будет доступен по адресу: http://localhost:5000"
echo "============================================================"
echo ""

# Запускаем MLflow UI
mlflow ui --backend-store-uri "file://$MLRUNS_DIR" --port 5000

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Ошибка при запуске MLflow UI"
    echo "Убедитесь, что MLflow установлен: pip install mlflow"
    exit 1
fi

