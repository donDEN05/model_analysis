import os
import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification




# Настройка путей (используем относительные пути)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

try:
    data = pd.read_csv(DATA_PATH)
    print(f'✅ Датасет загружен')
    
    # Проверка наличия необходимых колонок
    required_columns = ['text', 'emotion']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"❌ Ошибка: Отсутствуют необходимые колонки: {missing_columns}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Ошибка при чтении файла: {str(e)}")
    sys.exit(1)


device = "cpu"
use_cuda_env = torch.cuda.is_available()

# Детальная проверка доступности CUDA
cuda_available = torch.cuda.is_available()
if cuda_available:
    cuda_device_count = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0)
    cuda_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    print("="*60)
    print("🔍 Проверка CUDA:")
    print(f"   CUDA доступна: ✅")
    print(f"   Количество GPU: {cuda_device_count}")
    print(f"   Устройство 0: {cuda_device_name}")
    print(f"   Общая память GPU: {cuda_memory_total:.2f} GB")
    print(f"   Версия CUDA: {torch.version.cuda}")
    print(f"   Версия cuDNN: {torch.backends.cudnn.version()}")
    print("="*60)
    
    # Если USE_CUDA явно установлен в "false", используем CPU
    if use_cuda_env == False:
        device = "cpu"
        print("⚠️  CUDA доступна, но принудительно используется CPU (USE_CUDA=false)")
    else:
        device = 'cuda'
        print(f"✅ Используется GPU: {cuda_device_name}")
else:
    print("="*60)
    print("🔍 Проверка CUDA:")
    print("   CUDA доступна: ❌")
    print("   Причина: GPU не обнаружен или CUDA не установлена")
    print("="*60)
    print("ℹ️  Используется CPU (GPU не обнаружен)")
    print("   Для использования GPU убедитесь, что:")
    print("   - Установлены драйверы NVIDIA")
    print("   - Установлен PyTorch с поддержкой CUDA")
    print("   - GPU поддерживает CUDA")
    print("="*60)


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")