import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, Dropout, Input
from tensorflow.keras.models import Model

# Przykładowe parametry modelu
d_model = 128  # Wymiar modelu
num_heads = 4  # Liczba głów uwagi
dff = 512      # Rozmiar warstw feed-forward
input_vocab_size = 5000  # Rozmiar słownika wejściowego
target_vocab_size = 5000  # Rozmiar słownika wyjściowego
max_seq_len = 40          # Maksymalna długość sekwencji

# Tworzenie maski pozycyjnej dla uwagi
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

# Implementacja bloku enkodera
class SimpleEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_len, rate=0.1):
        super(SimpleEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

# Funkcja do tworzenia kodowania pozycyjnego
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

# Tworzenie modelu
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_seq_len, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = SimpleEncoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_len, rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        final_output = self.final_layer(enc_output)  # (batch_size, inp_seq_len, target_vocab_size)
        return final_output

# Inicjalizacja modelu
num_layers = 2  # Liczba warstw enkodera
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_seq_len)

# Przykładowe dane
input_seq = np.random.randint(0, input_vocab_size, (64, max_seq_len))
enc_padding_mask = create_padding_mask(input_seq)

# Kompilacja i trenowanie modelu
transformer.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = transformer.fit(input_seq, input_seq, epochs=5, batch_size=64)

# Test modelu
output = transformer(input_seq, training=False, enc_padding_mask=enc_padding_mask)
print(output.shape)  # Powinno zwrócić (batch_size, input_seq_len, target_vocab_size)
