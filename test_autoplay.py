import gradio as gr

def text_to_audio(text):
    # Simple function that returns a test audio file
    return "test_audio.wav"

# Create a simple Gradio interface to test autoplay
with gr.Blocks() as demo:
    gr.Markdown("# Autoplay Test")
    
    with gr.Row():
        text_input = gr.Textbox(label="Enter text")
        submit_btn = gr.Button("Submit")
    
    with gr.Row():
        audio_output = gr.Audio(label="Audio Output", type="filepath", autoplay=True)
    
    submit_btn.click(
        fn=text_to_audio,
        inputs=text_input,
        outputs=audio_output
    )

if __name__ == "__main__":
    # Create a test audio file
    with open("test_audio.wav", "wb") as f:
        # Write a simple WAV header and some data
        # This is a minimal WAV file with 1 second of silence
        f.write(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    
    demo.launch(debug=True)