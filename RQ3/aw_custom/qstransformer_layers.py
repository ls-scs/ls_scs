# Reimplementing the transformer block from https://keras.io/examples/nlp/text_classiication_with_transformer/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# The transformer block as a layer

class MultiHeadAttentionBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        # self.return_state = return_state
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, value, training, key=None):
        query = inputs
        if key==None:
            key=value
        attn_output = self.att(query, value, key)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm(query+attn_output)
        return out1

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, return_state=False, **kwargs):
        super(EncoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.return_state = return_state
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation='relu'),
                layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, value, training, key=None):
        query = inputs
        # print(training)
        if key==None:
            key=value
        attn_output = self.att(query, value, key)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm(query+attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        if self.return_state:
            #return self.layernorm(out1+ffn_output) + list(ffn_output[-1])
            return list(ffn_output[-1])
        else:
            return self.layernorm(out1+ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim' : self.ff_dim,
            'return_state': self.return_state
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, return_state=False, **kwargs):
        super(DecoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.return_state = return_state
        self.att1 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.att2 = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation='relu'),
                layers.Dense(self.embed_dim),
            ]
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, value, training, key=None):
        query = inputs
        # print(training)
        if key==None:
            key=value
        attn1_output = self.att1(query, query, query)
        attn1_output = self.dropout(attn1_output, training=training)
        out1 = self.layernorm(query+attn1_output)
        attn2_output = self.att2(out1, value, key)
        attn2_output = self.dropout(attn2_output, training=training)
        out2 = self.layernorm(out1+attn2_output)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout(ffn_output, training=training)
        if self.return_state:
            #return self.layernorm(out1+ffn_output) + list(ffn_output[-1])
            return list(ffn_output[-1])
        else:
            return self.layernorm(out2+ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim' : self.ff_dim,
            'return_state': self.return_state
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

# The embedding layer

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)

    def call(self, inputs):
        x = inputs
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x+positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim' : self.embed_dim
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape + (self.pos_emb.output_dim,)
