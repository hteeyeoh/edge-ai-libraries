# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gradio as gr
import logging
import requests
import os
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APP_BACKEND_URL = os.getenv("APP_BACKEND_URL", "http://localhost:9000/v1/eval")

# Define the sample markdown string
SAMPLE_MARKDOWN = """
**Instructions:** To use this application, please provide a markdown file with two sections titled 'Reference' and 'Generated', as shown in the example below.

### Example Markdown Format

```markdown
## Reference
<Your reference text goes here.>

## Generated
<Your machine-generated text goes here.>
```
---
"""


# Map dropdown options to API endpoints
METRIC_ENDPOINTS = {
    "bert-score": f"{APP_BACKEND_URL}/bert-score",
    "semantic-score": f"{APP_BACKEND_URL}/semantic-score",
    "rouge-score": f"{APP_BACKEND_URL}/rouge-score",
    "factual-consistency": f"{APP_BACKEND_URL}/factual-entailment"
}

METRIC_DESC= {
    "bert-score": "BERTScore leverages pre-trained contextual embeddings from BERT and similar models to evaluate text generation quality. It captures semantic similarity between texts by comparing their token-level embeddings, providing a more nuanced assessment than traditional metrics.",
    "semantic-score": "Captures meaning alignment beyond exact words; handles paraphrasing and synonyms.",
    "rouge-score": "Recall-Oriented Understudy for Gisting Evaluation. Measures how much of the important content/events are captured in the summary.",
    "factual-consistency": "Determines logical relationships between two text pieces. Evaluates how texts relate to each other."
}


def show_metric_description(choice):
    """Function to display the description for the selected option."""
    return METRIC_DESC.get(choice, "No description available.")

def submit_file(file):
    if file is None:
        return "Please upload a .md file."

    # Validate file format
    file_ext = os.path.splitext(file.name)[1].lower()
    if file_ext not in [".md"]:
        return "Unsupported file format. Please upload a .md file."

    eval_endpoint = f"{APP_BACKEND_URL}/evaluate"
    try:
        with open(file.name, "rb") as f:
            response = requests.post(eval_endpoint, files={"file": (file.name, f, "multipart/form-data")})

        if response.status_code == 200:
            data = response.json()

            # Separate last item as summary if it contains the summary key
            if isinstance(data, list) and len(data) > 0 and "Factual Consistency Summary" in data[-1]:
                summary = data[-1]["Factual Consistency Summary"]
                comparisons = data[:-1]
            else:
                summary = {}
                comparisons = data

            df = pd.DataFrame(comparisons)
            df.insert(0, "No.", range(1, len(df) + 1))

            summary_text = (
                "| Metric | Value |\n|---|---|\n" +
                "\n".join([f"| {k} | {v} |" for k, v in summary.items()])
            ) if isinstance(summary, dict) else str(summary)

            title_header = "### Factual Consistency Summary"

            return df, summary_text, title_header

        else:
            return f"Error: {response.status_code}: {response.text}"

    except Exception as e:
        return f"Request failed: {str(e)}"


def evaluate_metrics(reference, generated, metric):
    if not reference or not generated:
        return "Please enter both reference and generated text."

    metric_endpoint = METRIC_ENDPOINTS.get(metric)
    if not metric_endpoint:
        return "Invalid metric selected."

    payload = {
        "reference": reference,
        "generated": generated
    }

    try:
        response = requests.post(metric_endpoint, json=payload)

        if response.status_code == 200:
            result = response.json()

            # Format the result as a readable string
            formatted = "\n".join([f"{key.capitalize()}: {value}" for key, value in result.items()])

            return formatted

    except Exception as e:
        return f"Request failed: {str(e)}"



def create_ui():
    with gr.Blocks(title="Video-Accuracy Evaluation") as demo:
        gr.Markdown("# Video-Accuracy Evaluation Tool")
        gr.Markdown("Upload a .md file containing machine-generated and ground truth reference data for evaluation.")

        with gr.Tabs():
            with gr.TabItem("Summary Evaluation"):
                gr.Markdown(
                    "This tool evaluates factual consistency between a reference summary and a generated summary. "
                    "It compares sentences using similarity and Natural Language Inference (NLI), and provides detailed results and an overall summary."
                )
                gr.Markdown(SAMPLE_MARKDOWN)
                # Input file upload component
                file_input = gr.File(
                    label="Upload a .md file",
                    file_types=[".md"],
                    file_count="single"
                )
                submit_button = gr.Button("Submit")

                # Output display component
                result_table = gr.Dataframe(label="Sentence Comparisons", interactive=False)

                result_title = gr.Markdown()
                output_summary = gr.Markdown()

                # Set up button click event
                submit_button.click(
                    fn=submit_file,
                    inputs=[file_input],
                    outputs=[result_table, output_summary, result_title]
                )

            with gr.TabItem("Metrics"):
                gr.Markdown("This section allow user to quickly understand what are the avaiable metrics provided.")

                with gr.Row():
                    with gr.Column(scale=1):
                        reference_input = gr.Textbox(label="Reference Text", lines=4, placeholder="Enter reference text here...")

                    with gr.Column(scale=2):
                        generated_input = gr.Textbox(label="Generated Text", lines=4, placeholder="Enter generated text here...")

                metric_selector = gr.Dropdown(
                    choices=["bert-score", "semantic-score", "rouge-score", "factual-consistency"],
                    label="Select evaluation metric",
                    info="Available Metrics Provided for Evaluation",
                    value="bert-score",
                    interactive=True
                )

                metric_desc = gr.Markdown("Select a metric to understand more.")

                metric_submit_button = gr.Button("Submit")

                metric_output = gr.Textbox(label="Metric Result", lines=10)

                # Set up dropdown click event
                metric_selector.select(
                    fn=show_metric_description,
                    inputs=metric_selector,
                    outputs=metric_desc
                )

                # Set up button click event
                metric_submit_button.click(
                    fn=evaluate_metrics,
                    inputs=[reference_input, generated_input, metric_selector],
                    outputs=[metric_output]
                )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    logger.info(f"Starting Gradio UI...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=True
    )
