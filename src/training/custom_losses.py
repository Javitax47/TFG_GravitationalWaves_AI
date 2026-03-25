# src/training/custom_losses.py

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class MaskedHuberLoss(tf.keras.losses.Loss):
    """
    Calcula la Huber Loss pero ignora (enmascara) las muestras donde 
    la masa real suma 0.0 (es decir, las muestras que son puro Ruido).
    Esto evita que los gradientes se corrompan en arquitecturas Multi-Task.
    """
    def __init__(self, mask_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = mask_value
        # Usamos reduction NONE para obtener el error individual de cada muestra del lote
        self.huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        # 1. Calculamos el error bruto
        loss = self.huber(y_true, y_pred)
        
        # 2. Detectamos si es Ruido. Como y_true tiene [Masa1, Masa2],
        # si la suma absoluta es menor o igual a mask_value (0.0), es ruido.
        is_valid = tf.reduce_sum(tf.abs(y_true), axis=-1) > self.mask_value
        
        # 3. Convertimos a 1.0 (Señal) o 0.0 (Ruido)
        mask = tf.cast(is_valid, dtype=loss.dtype)
        
        # 4. Multiplicamos la pérdida por la máscara (el ruido se vuelve 0.0)
        masked_loss_values = loss * mask
        
        # 5. Calculamos la media dividiendo SOLO entre las señales válidas
        num_valid_samples = tf.reduce_sum(mask)
        
        # Sumamos 1e-7 para evitar el error de división por cero si en un lote casualmente todo es ruido
        return tf.reduce_sum(masked_loss_values) / (num_valid_samples + 1e-7)
        
    def get_config(self):
        config = super().get_config()
        config.update({"mask_value": self.mask_value})
        return config