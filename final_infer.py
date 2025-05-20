import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from peft import PeftModel
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run VQA evaluation")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to image dataset directory")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV metadata file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save output CSV")
    args = parser.parse_args()

    DATASET_DIR = args.dataset_dir
    CSV_PATH = args.csv_path
    OUTPUT_CSV = args.output_csv

    HF_REPO = "Hyma067/blip-vqa-lora-finetuned"
    BASE_MODEL = "Salesforce/blip-vqa-base"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor
    processor = BlipProcessor.from_pretrained(BASE_MODEL)

    # Load base model and apply LoRA adapter
    base_model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL).to(device)
    model = PeftModel.from_pretrained(base_model, HF_REPO).to(device)
    model.eval()

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df=df.head(100)
    def compute_bleu(reference, hypothesis):
        reference = [reference.lower().split()]
        hypothesis = hypothesis.lower().split()
        smoothie = SmoothingFunction().method4
        return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)

    def format_answer(answer):
        if not answer or pd.isna(answer):
            return "error"
        answer = str(answer).strip().split()[0]
        return answer.lower()

    answers = []
    bleus = []
    predictions = []
    references = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image_path = os.path.join(DATASET_DIR, row['path'])
            question = row['Question']
            expected_answer = str(row['Answer'])

            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, question, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs)

            answer = processor.decode(output[0], skip_special_tokens=True)
            answer_word = format_answer(answer)

        except Exception as e:
            print(f"Error processing {row.get('path', 'unknown')}: {str(e)}")
            answer_word = "error"

        answers.append(answer_word)
        predictions.append(answer_word)
        references.append(expected_answer)

        bleu = compute_bleu(expected_answer, answer_word)
        bleus.append(bleu)

    # BERTScore calculation (can be slow)
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)

    results_df = df.copy()
    results_df['generated_answer'] = answers
    results_df['bleu'] = bleus
    results_df['bertscore_f1'] = F1.tolist()

    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to {OUTPUT_CSV}")
    print("Average BLEU:", sum(bleus) / len(bleus))
    print("Average BERTScore F1:", F1.mean().item())

    print("\nSample results:")
    print(results_df[['Question', 'Answer', 'generated_answer', 'bleu', 'bertscore_f1']].head())

if __name__ == "__main__":
    main()
