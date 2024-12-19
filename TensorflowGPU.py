import tensorflow as tf

# 檢查 TensorFlow 版本
print("TensorFlow 版本:", tf.__version__)

# 檢查 GPU 是否可用
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# 列出所有 GPU 裝置
print("Physical Devices:", tf.config.experimental.list_physical_devices('GPU'))
