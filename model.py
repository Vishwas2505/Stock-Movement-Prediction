from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate
from transformers import TFBertModel

def build_model():

    # GRU branch
    price_input = Input(shape=(60,1))
    x1 = GRU(64)(price_input)
    price_feat = Dense(32, activation='relu')(x1)

    # BERT inputs (IMPORTANT FIX)
    input_ids = Input(shape=(32,), dtype='int32')
    attention_mask = Input(shape=(32,), dtype='int32')

    bert = TFBertModel.from_pretrained('bert-base-uncased')
    bert_output = bert(input_ids=input_ids, attention_mask=attention_mask)[1]

    text_feat = Dense(32, activation='relu')(bert_output)

    # HIMM
    combined = Concatenate()([price_feat, text_feat])
    x = Dense(64, activation='relu')(combined)
    x = Dense(32, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(
        inputs=[price_input, input_ids, attention_mask],
        outputs=output
    )

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model