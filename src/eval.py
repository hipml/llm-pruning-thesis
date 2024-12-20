import argparse
import evaluate
import gc
import json
import logging
import os
import pandas as pd
import psutil
import re
import string
import time
import torch
import types
import uuid
from codecarbon import OfflineEmissionsTracker
from datasets import load_dataset
from datetime import datetime
from pathlib import Path
from peft import get_peft_model, LoraConfig, PeftModel, TaskType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers import TrainerCallback
from transformers.cache_utils import Cache, DynamicCache
from torch import nn
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from rouge_score import rouge_scorer
from support.peconfig import PEConfig
from typing import Dict, Any

def setup_emissions_tracker(project_name="boolq_eval"):
    """
    Configure and create an emissions tracker instance.
    
    Returns:
        EmissionsTracker: Configured tracker instance
    """
    # experiment_id = uuid.uuid4()
    tracker = OfflineEmissionsTracker(
        project_name=project_name,
        tracking_mode='machine',
        measure_power_secs=120, 
    )
    return tracker

def save_results(results, config, n, adapter=True):
    """ Saves evaluation results to a JSON file  """

    os.makedirs(config.eval_dir, exist_ok=True)
    
    model_name = os.path.basename(config.formatted_name)
    adapter_str = "adapter" if adapter else "base"
    filename = f"{model_name}_{config.task}_{n}_{adapter_str}.json"
    
    # Combine all metadata with results
    full_results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_name": config.model_name,
            "num_pruned_layers": n,
            "has_adapter": adapter,
            "task": config.task
        },
        "metrics": results
    }
    
    # Save to file
    output_path = os.path.join(config.eval_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"Results saved to: {output_path}")

def evaluate_boolq(model, tokenizer, batch_size=16):
    """ Evaluates model performance on BoolQ dataset, tracking invalid responses. """
    # dataset = load_dataset("boolq", split="validation")
    dataset = load_dataset('hassansh/boolq_n_shot', '0_shot', split='test')
    model.eval()
    
    all_predictions = []
    all_labels = []
    invalid_responses = []  # store the actual text of invalid responses
    raw_predictions = [] 
    perplexities = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_data = dataset[i:i + batch_size]

        inputs = [f'Read the following passage and then answer the question as YES or NO or TRUE or FALSE only.\n{input_text}' for input_text in batch_data['input']]

        encoded_inputs = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=1,
                num_return_sequences=1,
                do_sample=False,
                top_p=None,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # calculate perplexity
            model_outputs = model(**encoded_inputs)
            logits = model_outputs.logits

            # prepare labels for loss calculation (shift by 1)
            labels = encoded_inputs['input_ids']
            labels = labels[:, 1:].contiguous()
            shift_logits = logits[:, :-1, :].contiguous()

            # calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)

        input_lengths = [len(ids) for ids in encoded_inputs.input_ids]
        generated_tokens = [
            output[input_len:] for output, input_len in zip(outputs, input_lengths)
        ]

        # Decode only the generated part
        predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        raw_predictions.extend(predictions)

        true_indicators = {'true', 'yes', 't', 'correct', 'right', 'indeed', 'absolutely', 'is', 'does', 'do'}
        false_indicators = {'false', 'no', 'f', 'incorrect', 'wrong', 'nope', 'negative', 'not', 'isn\'t', 'doesn\'t', 'don\'t' }

        def is_in(response_string, search_words):
            response_lower = response_string.lower().strip()
            return any(word in response_lower for word in search_words)

        for pred, label, question in zip(predictions, batch_data['target_str'], batch_data['input']):
            if is_in(pred, true_indicators):
                pred_answer = 1
            elif is_in(pred, false_indicators):
                pred_answer = 0
            else:
                words = pred.translate(str.maketrans('', '', string.punctuation)).lower().split()
                # print(f"pred: {pred} \t\t [T: {true_count} | F: {false_count}]")
                if words and words[0] in true_indicators:
                    pred_answer = 1
                elif words and words[0] in false_indicators:
                    pred_answer = 0
                else:
                    true_count = sum(1 for word in words if word in true_indicators)
                    false_count = sum(1 for word in words if word in false_indicators)
                    
                    if true_count > false_count:
                        pred_answer = 1
                    elif false_count > true_count:
                        pred_answer = 0
                    else:
                        pred_answer = -1

            # store all predictions (including None) and labels
            print(f"prediction: {pred} | label: {label} | fr: {pred_answer}", flush=True)
            all_predictions.append(pred_answer)
            all_labels.append(label)
            
            # store invalid responses with their questions for analysis
            if pred_answer == -1:
                invalid_responses.append({
                    'question': question,
                    'response': pred
                })
    
    # convert label from Yes/No to 1/0
    all_labels = [1 if label == 'Yes' else 0 for label in all_labels]
    # calculate total responses and invalid percentage
    total_responses = len(all_predictions)
    # invalid_count = sum(1 for pred in all_predictions if pred is None)
    invalid_count = len(invalid_responses)
    valid_count = total_responses - invalid_count
    invalid_percentage = (invalid_count / total_responses * 100) if total_responses > 0 else 0
    
    # Calculate metrics only on valid responses
    valid_predictions = [p for p in all_predictions if p != -1]
    valid_labels = [l for p, l in zip(all_predictions, all_labels) if p != -1]
    
    # Calculate confusion matrix metrics (only for valid responses)
    true_positives = sum((p == 1 and l == 1) for p, l in zip(valid_predictions, valid_labels))
    true_negatives = sum((p == 0 and l == 0) for p, l in zip(valid_predictions, valid_labels))
    false_positives = sum((p == 1 and l == 0) for p, l in zip(valid_predictions, valid_labels))
    false_negatives = sum((p == 0 and l == 1) for p, l in zip(valid_predictions, valid_labels))
    
    # Calculate standard metrics (only for valid responses)
    accuracy = (true_positives + true_negatives) / valid_count if valid_count > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    accuracy_metric = evaluate.load("accuracy")
    accuracy_results = accuracy_metric.compute(references=all_labels, predictions=all_predictions)
    print(f"Evaluate: {accuracy_results}")

    results = {
        "response_quality": {
            "total_responses": total_responses,
            "valid_responses": valid_count,
            "invalid_responses": invalid_count,
            "invalid_percentage": invalid_percentage
        },
        "metrics": {
            "accuracy": accuracy,
            "evaluate_accuracy": accuracy_results,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_evaluated": valid_count,
            "confusion_matrix": {
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            },
            "perplexities": perplexities,
            "mean_perplexity": sum(perplexities) / len(perplexities)
        },
        "invalid_examples": invalid_responses[:10]  
    }
    
    return results

def evaluate_summarization(model, tokenizer, batch_size=16) -> Dict[str, Any]:
    dataset = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
    model.eval()
    
    all_summaries = []
    all_references = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i + batch_size]

        prompt = "Summarize this document in one sentence:"
        prompt2 = "Your summary:"

        inputs = tokenizer(
            [prompt+ doc + prompt2 for doc in batch["document"]],
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            # Generate summaries
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=1,
                min_length=8,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode summaries
        input_lengths = [len(ids) for ids in inputs.input_ids]
        generated_tokens = [
            output[input_len:] for output, input_len in zip(outputs, input_lengths)
        ]
        summaries = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        first_sentences = [re.split(r'(?<=[.!?])\s*', summary.strip())[0] for summary in summaries]
        for x in first_sentences:
            print(x, flush=True)
        print("~~~", flush=True)

        all_summaries.extend(first_sentences)
        # all_summaries.extend(summaries)
        all_references.extend(batch["summary"])
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {
        'rouge1': [], 'rouge2': [], 'rougeL': []
    }
    
    for pred, ref in zip(all_summaries, all_references):
        score = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key].append(score[key].fmeasure)
    
    # Calculate average scores
    avg_rouge = {k: torch.tensor(v, dtype=torch.float).mean().item() for k, v in rouge_scores.items()}

    results = {
        "metrics": {
            "rouge": avg_rouge
        },
        "examples": {
            "predictions": all_summaries[:5],
            "references": all_references[:5]
        },
        "summary": {
            "total_samples": len(all_summaries),
            "avg_summary_length": torch.tensor([len(s.split()) for s in all_summaries], dtype=torch.float).mean().item()
        }
    }    

    return results

def evaluate_imdb(model, tokenizer, batch_size=16):
    """ Evaluates model performance on IMDB dataset, tracking invalid responses. """
    dataset = load_dataset("stanfordnlp/imdb", split="test")
    model.eval()
    
    all_predictions = []
    all_labels = []
    invalid_responses = []
    raw_predictions = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_data = dataset[i:i + batch_size]

        inputs = [f'{review}\nReview was POSITIVE or NEGATIVE:' 
                 for review in batch_data['text']]

        encoded_inputs = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **encoded_inputs,
                max_new_tokens=2,
                num_return_sequences=1,
                do_sample=False,
                top_p=None,
                temperature=None,
                min_length=10,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_lengths = [len(ids) for ids in encoded_inputs.input_ids]
        generated_tokens = [
            output[input_len:] for output, input_len in zip(outputs, input_lengths)
        ]

        predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        raw_predictions.extend(predictions)

        positive_indicators = {'positive', 'good', 'great', 'excellent', 'pos', 'like', 'loved', 'amazing'}
        negative_indicators = {'negative', 'bad', 'poor', 'terrible', 'neg', 'dislike', 'hated', 'awful'}

        def is_in(response_string, search_words):
            response_lower = response_string.lower().strip()
            return any(word in response_lower for word in search_words)

        for pred, label, review in zip(predictions, batch_data['label'], batch_data['text']):
            if is_in(pred, positive_indicators):
                pred_answer = 1
            elif is_in(pred, negative_indicators):
                pred_answer = 0
            else:
                words = pred.translate(str.maketrans('', '', string.punctuation)).lower().split()
                if words and words[0] in positive_indicators:
                    pred_answer = 1
                elif words and words[0] in negative_indicators:
                    pred_answer = 0
                else:
                    positive_count = sum(1 for word in words if word in positive_indicators)
                    negative_count = sum(1 for word in words if word in negative_indicators)
                    
                    if positive_count > negative_count:
                        pred_answer = 1
                    elif negative_count > positive_count:
                        pred_answer = 0
                    else:
                        pred_answer = -1

            print(f"prediction: {pred} | label: {label} | parsed: {pred_answer}", flush=True)
            all_predictions.append(pred_answer)
            all_labels.append(label)
            
            if pred_answer == -1:
                invalid_responses.append({
                    'review': review[:100] + "...",  # First 100 chars of review
                    'response': pred
                })
    # transform labels to integers ("0 neg" -> 0, "1 pos" -> 1)
    # all_labels = [int(l[0]) for l in all_labels]
    
    total_responses = len(all_predictions)
    invalid_count = len(invalid_responses)
    valid_count = total_responses - invalid_count
    invalid_percentage = (invalid_count / total_responses * 100) if total_responses > 0 else 0
    
    valid_predictions = [p for p in all_predictions if p != -1]
    valid_labels = [l for p, l in zip(all_predictions, all_labels) if p != -1]
    
    true_positives = sum((p == 1 and l == 1) for p, l in zip(valid_predictions, valid_labels))
    true_negatives = sum((p == 0 and l == 0) for p, l in zip(valid_predictions, valid_labels))
    false_positives = sum((p == 1 and l == 0) for p, l in zip(valid_predictions, valid_labels))
    false_negatives = sum((p == 0 and l == 1) for p, l in zip(valid_predictions, valid_labels))
    
    accuracy = (true_positives + true_negatives) / valid_count if valid_count > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    accuracy_metric = evaluate.load("accuracy")
    accuracy_results = accuracy_metric.compute(references=all_labels, predictions=all_predictions)
    print(f"Evaluate: {accuracy_results}")

    results = {
        "response_quality": {
            "total_responses": total_responses,
            "valid_responses": valid_count,
            "invalid_responses": invalid_count,
            "invalid_percentage": invalid_percentage
        },
        "metrics": {
            "accuracy": accuracy,
            "evaluate_accuracy": accuracy_results,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_evaluated": valid_count,
            "confusion_matrix": {
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            },
        },
        "invalid_examples": invalid_responses[:10]
    }
    
    return results

def prune_model(model, config, n):
    """Prunes n consecutive layers from the model and updates KV cache indexing."""
    if n == 0:
        return model
        
    # read and filter similarity CSV
    df = pd.read_csv(config.csv_path)
    filtered_df = df[
        (df['layers_to_prune'] == n) & 
        (df['metric'] == 'exp')
    ].sort_values('similarity', ascending=False)
    
    if filtered_df.empty:
        raise ValueError(f"No matching entries found for {n} layers to prune")
    
    # get starting layer index and layers to prune
    start_layer = filtered_df.iloc[0]['starting_layer']
    layers_to_prune = range(start_layer, start_layer + n)
    
    # Update model config
    model.config.num_hidden_layers -= n
    
    # Prune layers
    model.model.layers = nn.ModuleList([
        layer for i, layer in enumerate(model.model.layers)
        if i not in layers_to_prune
    ])
    
    # Store which layers were pruned for KV cache handling
    model.pruned_layers = list(layers_to_prune)
    
    # Modify forward pass to handle KV cache
    original_forward = model.forward
    
    def new_forward(self, *args, **kwargs):
        if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
            past_kv = kwargs['past_key_values']
            kwargs['past_key_values'] = tuple(
                kv for i, kv in enumerate(past_kv) 
                if i not in self.pruned_layers
            )
        return original_forward(*args, **kwargs)
    
    model.forward = types.MethodType(new_forward, model)
    
    print(f"Pruned architecture: \n{model}")
    return model

def load_adapter(model, config):
    try:
        print(f"Loading adapter from {config.adapter_path}...")
        adapter_model = PeftModel.from_pretrained(model, config.adapter_path)
        print("Success")
        return adapter_model, True
    except:
        print("Failed to load adapter")
        return model, False

def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.auth_token,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left"
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
 

def load_model(model_name, auth_token):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=auth_token,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config
    )
    return model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Distributed evaluation of pruned LLM")
    parser.add_argument("--model", type=str, required=True, help="Model identifier from config")
    parser.add_argument("--task", type=str, required=True, help="Task identifier from config")
    parser.add_argument("--n", type=int, required=True, help="Number of layers to prune")
    return parser.parse_args()

    
def main():
    args = parse_arguments()
    config = PEConfig(args)

    tokenizer = load_tokenizer(config)
    model = load_model(config.model_name, config.auth_token)

    model = prune_model(model, config, args.n)

    task_map = {
        "boolq": evaluate_boolq,
        "summ": evaluate_summarization,
        "imdb": evaluate_imdb,
    }
    
    if args.task not in task_map:
        raise ValueError(f"Unknown task: {args.task}")
    
    # eval the pruned model
    project_name = f'{config.model_name}_eval_{config.task}_base'
    tracker = setup_emissions_tracker(project_name=project_name)
    tracker.start()
    results = task_map[args.task](model, tokenizer)
    # no_adapter_results = evaluate_boolq(model, tokenizer)
    emissions = tracker.stop()
    print(emissions)
    
    save_results(results, config, args.n, adapter=False)

    model, found_adapter = load_adapter(model, config)
    if not found_adapter:
        print("Healed model not found!")
        return 1

    # eval the pruned model with adapter
    project_name = f'{config.model_name}_eval_{config.task}_adapter'
    tracker = setup_emissions_tracker(project_name=project_name)
    tracker.start()
    adapter_results = task_map[args.task](model, tokenizer)
    emissions = tracker.stop()
    print(emissions)
    
    save_results(adapter_results, config, args.n, adapter=True)

if __name__ == "__main__":
    main()
