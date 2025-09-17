from .config import config
from .logger import logger
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any
import csv
import markdown
import numpy as np
import os
import nltk
import torch


class Evaluator:
    def __init__(self, sbert_model_name: str = None, nli_model_name: str = None):
        """
        Initialize the evaluator class.

        Args:
            sbert_model_name: Sentence transformer model name
            nli_model_name: Natural Language Inference model name
        """

        # Download necessary NLTK resources
        nltk.download('punkt')
        nltk.download('punkt_tab')

        # Initialize the Sentence Transformer model
        self.sentence_model = SentenceTransformer(
            sbert_model_name,
            cache_folder=f"{config._CACHE_DIR}/{sbert_model_name}"
        )

        # Initialize the NLI model and tokenizer
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            nli_model_name,
            cache_dir=f"{config._CACHE_DIR}/{nli_model_name}"
        )

        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model_name,
            cache_dir=f"{config._CACHE_DIR}/{nli_model_name}"
        )


    def _calculate_semantic_score(self, generated: str="", reference: str="") -> Dict[str, Any]:
        """
        Calculate semantic similarity score between generated and reference text using Sentence-BERT embeddings.

        Args:
            generated (str): The generated text to be evaluated.
            reference (str): The reference text to compare against.

        Returns:
            dict: A dictionary containing precision, recall, F1 score, and similarity score.
        """

        logger.info(f"Calculating semantic score...")
        if not generated or not reference:
            raise ValueError("Both 'generated' and 'reference' must be non-empty strings.")

        # Tensor output
        embeddings = self.sentence_model.encode([generated, reference], convert_to_tensor=True)
        # NumPy output
        # embeddings = self.sentence_model.encode([generated, reference])
        similarity = float(util.cos_sim(embeddings[0], embeddings[1]).item())

        gen_len = len(generated.split())
        ref_len = len(reference.split())
        length_ratio = gen_len / ref_len if ref_len > 0 else 1

        precision = float(similarity / max(length_ratio, 1.0))
        recall = float(similarity * min(length_ratio, 1.0))
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "reference": reference,
            "generated": generated,
            "precision": round(min(precision, 1.0), 4),
            "recall": round(min(recall, 1.0), 4),
            "f1_score": round(f1, 4),
            "similarity": round(similarity, 4)
        }


    def _calculate_rouge_score(self, generated: str = "", reference: str = "") -> list[Dict[str, Any]]:
        """
        Calculate ROUGE-1, ROUGE-2 and ROUGE-L scores between generated and reference text.

        Args:
            generated (str): The generated text to be evaluated.
            reference (str): The reference text to compare against.

        Returns:
            dict: A dictionary containing precision, recall, and F1 scores for ROUGE-1, ROUGE-2, and ROUGE-L.
        """
        logger.info("Calculating ROUGE score...")
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generated)  # reference first, then generated

        return {
            "reference": reference,
            "generated": generated,
            "rouge1": {
                "precision": round(scores["rouge1"].precision, 4),
                "recall": round(scores["rouge1"].recall, 4),
                "f1_score": round(scores["rouge1"].fmeasure, 4)
            },
            "rouge2": {
                "precision": round(scores["rouge2"].precision, 4),
                "recall": round(scores["rouge2"].recall, 4),
                "f1_score": round(scores["rouge2"].fmeasure, 4)
            },
            "rougeL": {
                "precision": round(scores["rougeL"].precision, 4),
                "recall": round(scores["rougeL"].recall, 4),
                "f1_score": round(scores["rougeL"].fmeasure, 4)
            }
        }


    def _calculate_average_scores(self, pairs: list[tuple[str, str]]) -> list[Dict[str, Any]]:
        """
        Calculate average ROUGE and Semantic scores for multiple (generated, reference) pairs.

        Args:
            pairs (list of tuples): List of (generated, reference) text pairs.

        Returns:
            dict: Dictionary containing average precision, recall, and F1 scores for ROUGE-1, ROUGE-2, ROUGE-L, and Semantic metrics.
        """
        logger.info("Calculating average scores for multiple pairs...")
        # Holders for metrics
        rouge1_p, rouge1_r, rouge1_f = [], [], []
        rouge2_p, rouge2_r, rouge2_f = [], [], []
        rougel_p, rougel_r, rougel_f = [], [], []
        semantic_p, semantic_r, semantic_f, semantic_s = [], [], [], []

        for generated, reference in pairs:
            rouge_scores = self._calculate_rouge_score(generated, reference)
            semantic_scores = self._calculate_semantic_score(generated, reference)

            rouge1_p.append(rouge_scores["rouge1"]["precision"])
            rouge1_r.append(rouge_scores["rouge1"]["recall"])
            rouge1_f.append(rouge_scores["rouge1"]["f1_score"])

            rouge2_p.append(rouge_scores["rouge2"]["precision"])
            rouge2_r.append(rouge_scores["rouge2"]["recall"])
            rouge2_f.append(rouge_scores["rouge2"]["f1_score"])

            rougel_p.append(rouge_scores["rougeL"]["precision"])
            rougel_r.append(rouge_scores["rougeL"]["recall"])
            rougel_f.append(rouge_scores["rougeL"]["f1_score"])

            semantic_p.append(semantic_scores["precision"])
            semantic_r.append(semantic_scores["recall"])
            semantic_f.append(semantic_scores["f1_score"])
            semantic_s.append(semantic_scores["similarity"])

        # return average dictionary
        return {
            "rouge1": {
                "precision": round(np.mean(rouge1_p), 4),
                "recall": round(np.mean(rouge1_r), 4),
                "f1_score": round(np.mean(rouge1_f), 4),
            },
            "rouge2": {
                "precision": round(np.mean(rouge2_p), 4),
                "recall": round(np.mean(rouge2_r), 4),
                "f1_score": round(np.mean(rouge2_f), 4),
            },
            "rougeL": {
                "precision": round(np.mean(rougel_p), 4),
                "recall": round(np.mean(rougel_r), 4),
                "f1_score": round(np.mean(rougel_f), 4),
            },
            "semantic": {
                "precision": round(np.mean(semantic_p), 4),
                "recall": round(np.mean(semantic_r), 4),
                "f1_score": round(np.mean(semantic_f), 4),
                "similarity": round(np.mean(semantic_s), 4),
            }
        }


    def _evaluate_factual_consistency(self, generated: str="", reference: str="") -> str:
        """
        Evaluates the factual consistency between a generated sentence and a reference sentence using a Natural Language Inference (NLI) model.
        The function applies the NLI model to determine the relationship (entailment, neutral, contradiction) between the two sentences.

        Args:
            generated (str): The generated sentence to be evaluated.
            reference (str): The reference sentence to compare against.

        Returns:
            str: The NLI label indicating the relationship between the sentences: "entailment", "neutral", or "contradiction".
        """
        logger.info("Calculating factual consistency...")

        # Apply NLI model to best match sentence to find relationship
        # Convert text sentences into numerical tokens that model can understand
        inputs = self.nli_tokenizer(reference, generated, return_tensors="pt", truncation=True)
        # NLI models itself usually built using pyTorch, so we use torch.no_grad() to avoid computing gradients (saves memory)
        # Telling PyTorch: "we are in inference mode, not training mode"
        with torch.no_grad():
            # Passes the tokenized inputs through the NLI model to get raw prediction scores(logits) for each possible class
            logits = self.nli_model(**inputs).logits
            # Find which class has highest score
            predicted_class = logits.argmax(dim=1).item()
            # Map the predicted class index to human-readable label
            label = self.nli_model.config.id2label[predicted_class]

        return {
            "reference": reference,
            "generated": generated,
            "label": label
        }

    def _evaluate_summaries(self, reference: str="", generated: str="") -> list[Dict[str, Any]]:
        """
        Evaluates the factual consistency between a reference summary and a generated summary using sentence similarity and Natural Language Inference (NLI).
        The function compares each sentence in the reference summary to its most similar sentence in the generated summary using sentence embeddings and cosine similarity.
        It then applies an NLI model to determine the relationship (entailment, neutral, contradiction) between the matched sentences.
        The results include per-sentence comparison details and an aggregated summary of factual consistency statistics.

        Args:
            reference (str): The reference summary text to compare against.
            generated (str): The generated summary text to be evaluated.

        Returns:
            List[dict]: A list of dictionaries containing:
            - Per-sentence comparison details:
                "reference_sentence": str,
                "matched_sentence": str,
                "similarity_score": float,
                "nli_label": str
            - A summary dictionary with factual consistency statistics:
                    "total_sentences_compared": int,
                    "entailment": int,
                    "neutral": int,
                    "contradiction": int,
                    "entailment_ratio": float,
                    "neutral_ratio": float,
                    "contradiction_ratio": float
        """

        # Initialize counters
        entailment_count = 0
        contradiction_count = 0
        neutral_count = 0
        total_cosine_score = 0

        # Split the texts into sentences
        ref_sentences = sent_tokenize(reference)
        gen_sentences = sent_tokenize(generated)

        # Encode sentences from generated summary
        gen_embeddings = self.sentence_model.encode(gen_sentences, convert_to_tensor=True)

        # Compare each sentence in reference summary to best match in generated summary
        results = []
        for ref_sentence in ref_sentences:
            ref_embedding = self.sentence_model.encode(ref_sentence, convert_to_tensor=True)
            cosine_scores = util.cos_sim(ref_embedding, gen_embeddings)[0]
            best_match_idx = torch.argmax(cosine_scores).item()
            best_match_sentence = gen_sentences[best_match_idx]

            result = self._evaluate_factual_consistency(best_match_sentence, ref_sentence)
            label = result["label"]

            results.append(
                {
                    "reference_sentence": ref_sentence,
                    "matched_sentence":best_match_sentence,
                    "similarity_score": round(float(cosine_scores[best_match_idx].item()), 4),
                    "nli_label": label
                }
            )

            if label == "entailment":
                entailment_count += 1
            elif label == "neutral":
                neutral_count += 1
            else:
                contradiction_count += 1

            # Calculate total and proportions
            total = entailment_count + neutral_count + contradiction_count
            entailment_ratio = entailment_count / total if total > 0 else 0
            neutral_ratio = neutral_count / total if total > 0 else 0
            contradiction_ratio = contradiction_count / total if total > 0 else 0
            total_cosine_score += float(cosine_scores[best_match_idx].item())

        results.append({
            "Factual Consistency Summary": {
                "total_sentences_compared": total,
                "total_cosine_score": round(total_cosine_score, 4),
                "entailment": entailment_count,
                "neutral": neutral_count,
                "contradiction": contradiction_count,
                "entailment_ratio": round(entailment_ratio, 2),
                "neutral_ratio": round(neutral_ratio, 2),
                "contradiction_ratio": round(contradiction_ratio, 2),
                "average_cosine_score": round(total_cosine_score / total, 2) if total > 0 else 0
            }
        })

        return results


    def run_evaluation_from_file(self, file_path:str) -> list[Dict[str, Any]]:
        """
        Runs evaluation based on the file type (.tsv or .md) and returns the results.

        Args:
            file_path (str): Path to the input file (.tsv or .md).

        Returns:
            list of dict: Evaluation results containing metrics scores and other relevant information.
        """
        logger.info(f"Running evaluation from file: {file_path}")
        results = {
            "file": os.path.basename(file_path),
            "metrics_scores": []
        }
        with open(file_path, newline="", encoding="utf-8") as file:
            if file_path.lower().endswith(".tsv"):
                reader = csv.DictReader(file, delimiter="\t")
                for row in reader:
                    generated = row["Generated"]
                    reference = row["Reference"]
                    scores = self._calculate_average_scores([(generated, reference)])
                    results["metrics_scores"].append(scores)

                return results

            elif file_path.lower().endswith(".md"):
                md_text = file.read()

                # Convert markdown to HTML
                html = markdown.markdown(md_text)

                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                headers = soup.find_all("h2")

                sections = []
                for header in headers:
                    content = []
                    for sibling in header.find_next_siblings():
                        if sibling.name == "h2":
                            break
                        content.append(sibling.get_text())
                    sections.append("\n".join(content).strip())

                if len(sections) < 2:
                    raise ValueError("Markdown file must contain at least two sections for comparison.")

                results = self._evaluate_summaries(sections[0], sections[1])

                return results