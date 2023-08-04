import gradio as gr

from templates.utils import train_model, predict

with gr.Blocks() as train:
    with gr.Row():
        train_btn = gr.components.Button('Train', scale=2)
        output_text = gr.components.Textbox(label='Result', scale=4)
        output_json = gr.components.JSON(label='Metrics', scale=4)
    train_btn.click(
        fn=train_model,
        inputs=None,
        outputs=[output_text, output_json]
    )


inference = gr.Interface(
    fn=predict,
    inputs=[gr.components.Number(label='Fixed Acidity'), gr.components.Number(label='Volatile Acidity'), gr.components.Number(label='Citric Acid'),
            gr.components.Number(label='Residual Sugar'), gr.components.Number(label='Chlorides'), gr.components.Number(label='Free Sulfur Dioxide'),
            gr.components.Number(label='Total Sulfur Dioxide'), gr.components.Number(label='Density'), gr.components.Number(label='pH'),
            gr.components.Number(label='Sulphates'), gr.components.Number(label='Alcohol')],
    outputs=gr.components.Textbox(label='Prediction'),
    allow_flagging='never',
    description='Inference the modal.'
)

demo = gr.TabbedInterface([train, inference], ['Train', 'Inference'])
