import tensorflow as tf
import gradio as gr
import pandas as pd
import os

# Load the best performing model
best_model = tf.keras.models.load_model(os.path.join('model', 'best_model.keras'))

# Load vocabulary
vectorize_layer = tf.keras.layers.TextVectorization()
vocab_file = os.path.join('models')
vectorize_layer.load_assets(vocab_file)

# Load label mapping
label_mapping = pd.read_csv(os.path.join('model', 'label_mapping.csv'))

def predict(text):
    # Vectorize the input text
    text_vector = vectorize_layer([text])
    
    # Make prediction
    pred = best_model.predict(text_vector, verbose=0)
    
    # If pred is probabilities, get the index of the highest probability
    predicted_index = tf.argmax(pred, axis=1).numpy()[0]
    
    # Map index to label
    predicted_label = label_mapping[label_mapping['index'] == predicted_index]['label'].values[0]
    
    return predicted_label

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter your text", lines=3),
    outputs=gr.Textbox(label="Result"),
    title="My NLP Model",
    description="Enter text to process with my TensorFlow model"
)

if __name__ == "__main__":
    demo.launch()