########Muti-label classification model#######

# Import the required python packages
import pandas as pd
import numpy as np
import os
import random
from ast import literal_eval
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from keras.callbacks import EarlyStopping
from keras import optimizers
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

# Setting seed value for reproducibility
SEED = 42
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


# Make dataset function definition
def make_dataset(dataframe, batch_size, lookup, is_train=True):
    labels = tf.ragged.constant(dataframe["Original_Label"].values)
    label_binarized = lookup(labels).numpy()
    embed1=dataframe.embeddings.values
    temp=[list(i) for i in embed1]
    embed2=np.array(temp)
    dataset = tf.data.Dataset.from_tensor_slices(
         (embed2, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)

# Make dataset new function definition
def inference_data(dataframe, batch_size,lookup,is_train=True):
    labels = tf.ragged.constant(dataframe["Original_Label"].values)
    label_binarized = lookup(labels).numpy()
    embed1=dataframe.embeddings.values
    temp=[list(i) for i in embed1]
    summaries=dataframe.Abstract.values
    embed2=np.array(temp)
    dataset = tf.data.Dataset.from_tensor_slices(
         (embed2, label_binarized,summaries)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)

# Inverse multi-hot encoded label to tuple function definition
def invert_multi_hot(encoded_labels,vocab):
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)

# Make model definition
def make_model(lookup):
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(1024, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup.vocabulary_size(), activation="sigmoid"),
        ]  
    )
    return shallow_mlp_model

# Make inference definition    
def make_infernce(df,batch_size,shallow_mlp_model,lookup,vocab):
    # Create a model for inference.
    model_for_inference = keras.Sequential([shallow_mlp_model])
    inference_dataset = inference_data(df, batch_size, lookup,is_train=False)
    text_batch, label_batch,label_summary = next(iter(inference_dataset))
    predicted_probabilities = model_for_inference.predict(text_batch)
    df_final_pred = pd.DataFrame(columns = ['Abstract', 'Original_Label', 'Predicted_Label'])
    for i, text in enumerate(text_batch):
        label = label_batch[i].numpy()[None, ...]
        orig_lab=str(invert_multi_hot(label[0],vocab))
        orig_lab=orig_lab.strip("[']'").split("\' \'")
        top_3_labels = [
            x
            for _, x in sorted(
                zip(predicted_probabilities[i], lookup.get_vocabulary()),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ][:5]
        pred_lab=','.join([label for label in top_3_labels])
        new_row = {'Abstract':label_summary[i], 'Original_Label':orig_lab, 'Predicted_Label':pred_lab}

    # Append the new row to the DataFrame
        df_final_pred = pd.concat([df_final_pred, pd.DataFrame([new_row])], ignore_index=True)
    return df_final_pred

# Main block
if __name__ == "__main__":
    data=pd.read_excel("training_dataset_raw.xlsx")
    data_final=data.loc[data["Abstract"]!=0]
    data_final=data_final.reset_index(drop=True)
    data_filtered=data_final[["Abstract","Original_Label"]]

    data_filtered = data_filtered.groupby("Original_Label").filter(lambda x: len(x) > 1)

    data_filtered=data_filtered.reset_index(drop=True)
    data_filtered["Original_Label"] = data_filtered["Original_Label"].apply(
        lambda x: literal_eval(x)
    )
    model_sentence = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    data_filtered["embeddings"]=data_filtered.apply(lambda row: model_sentence.encode(row["Abstract"]),axis=1)
    data_filtered['embeddings'] = data_filtered['embeddings'].apply(lambda x: ','.join(map(str, x)))

    data_filtered['embeddings'] = data_filtered['embeddings'].apply(lambda x: list(map(float, str(x).split(','))))
    data_filtered['embeddings'] = data_filtered['embeddings'].apply(lambda x: np.array(x))
    data_filtered["Original_Label"] = data_filtered["Original_Label"].apply(
    lambda x: x.replace("\'","").split(',')
    )



    test_split = 0.2
    # Initial train and test split.
    train_df, test_df = train_test_split(
        data_filtered,
        test_size=test_split,
        stratify=data_filtered["Original_Label"].values,
    )

    # Splitting the test set further into validation and test sets.
    val_df = test_df.sample(frac=0.5)
    test_df.drop(val_df.index, inplace=True)

    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")

    terms = tf.ragged.constant(train_df["Original_Label"].values)
    lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
    lookup.adapt(terms)
    vocab = lookup.get_vocabulary()
    batch_size = 128
    epochs = 200

    labels = tf.ragged.constant(train_df["Original_Label"].values)
    label_binarized = lookup(labels).numpy()
    train_dataset = make_dataset(train_df, batch_size,lookup,is_train=True)
    validation_dataset = make_dataset(val_df, batch_size,lookup,is_train=False)
    test_dataset = make_dataset(test_df, batch_size,lookup,is_train=False)

    earlyStopping = EarlyStopping(monitor='val_binary_accuracy',
                                           restore_best_weights=True,
                                           patience=15, verbose=0, mode='max'  ,min_delta=0.0001
                                           )
    adam = optimizers.Adam(learning_rate=0.001, decay=0.0001)
    shallow_mlp_model = make_model(lookup)
    shallow_mlp_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
    )

    history = shallow_mlp_model.fit(
        train_dataset, validation_data=validation_dataset, epochs=epochs,callbacks=[earlyStopping]
    )

    _, binary_acc = shallow_mlp_model.evaluate(test_dataset)
    shallow_mlp_model.save("model.h5")
 
    test_df_predicted=make_infernce(test_df,len(test_df.index),shallow_mlp_model,lookup,vocab)
    test_df_predicted["Original_Label"] = test_df_predicted["Original_Label"].apply(
    lambda x: str(x).replace("\"","\'").replace("'\n '", ",").replace("[\'","").replace("\']","").replace("\', \'",",")
    )
    
    test_df_predicted["Original_Label"] = test_df_predicted["Original_Label"].apply(
    lambda x: str(x).replace("'\\n ' ", ",").replace("'\\n '", ",")
    )
    
    test_df_predicted['Original_Label'] = test_df_predicted['Original_Label'].apply(lambda x: x.strip("[]").replace("\'","").split(","))
    test_df_predicted['Predicted_Label'] = test_df_predicted['Predicted_Label'].apply(lambda x: x.split(","))

    mlb = MultiLabelBinarizer()
    original_labels = mlb.fit_transform(test_df_predicted['Original_Label'])
    predicted_labels = mlb.transform(test_df_predicted['Predicted_Label'])
   
   # Evaluation Metrics
 
    # Hamming loss
    hamming_loss = hamming_loss(original_labels, predicted_labels)

    # Micro average
    micro_precision = precision_score(original_labels, predicted_labels, average='micro')
    micro_recall = recall_score(original_labels, predicted_labels, average='micro')
    micro_f1 = f1_score(original_labels, predicted_labels, average='micro')

    # Print the results
    print("Precision (Micro):", micro_precision)
    print("Recall (Micro):", micro_recall)
    print("F1 Score (Micro):", micro_f1)
    print("Hamming Loss:", hamming_loss)
