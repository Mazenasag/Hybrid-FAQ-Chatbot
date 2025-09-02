import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List

class NeuralQuestionRephraser:
    def __init__(self):
        self.model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.generation_config = {
            "max_length": 128,
            "num_beams": 10,
            "num_beam_groups": 5,
            "num_return_sequences": 5,
            "diversity_penalty": 1.0,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95
        }

    def _clean_question(self, question: str) -> str:
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        return question[0].upper() + question[1:]

    def _postprocess_paraphrases(self, generated: List[str], original: str) -> List[str]:
        unique_paraphrases = set()
        for text in generated:
            text = text.strip()
            if text.endswith('</s>'):
                text = text[:-4].strip()
            if not text.endswith('?'):
                text += '?'
            text = text[0].upper() + text[1:]
            if (text.lower() != original.lower() and 
                not any(text.lower() == p.lower() for p in unique_paraphrases)):
                unique_paraphrases.add(text)
        return list(unique_paraphrases)

    def rephrase_question(self, original_question: str, num_variants: int = 3) -> List[str]:
        original_question = self._clean_question(original_question)
        input_text = f"paraphrase: {original_question}"
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors="pt",
            max_length=128,
            truncation=True
        )
        outputs = self.model.generate(
            input_ids,
            **self.generation_config
        )
        decoded_outputs = [
            self.tokenizer.decode(output, skip_special_tokens=False) 
            for output in outputs
        ]
        paraphrases = self._postprocess_paraphrases(decoded_outputs, original_question)
        return paraphrases[:num_variants]

# Load your Excel file
df = pd.read_csv("/content/ecommerce_faq_60_unique.csv")  # Replace with your actual file path

rephraser = NeuralQuestionRephraser()

# Prepare new dataframe rows
new_rows = []

for idx, row in df.iterrows():
    group_id = idx + 1  # Group ID starts at 1
    original_q = row['Question']
    answer = row['Answer']
    qtype = row['Type']
    
    # Add original question row
    new_rows.append({
        "GroupID": group_id,
        "Question": original_q,
        "Answer": answer,
        "Type": qtype
    })
    
    # Generate paraphrases
    paraphrases = rephraser.rephrase_question(original_q, num_variants=3)
    for para in paraphrases:
        new_rows.append({
            "GroupID": group_id,
            "Question": para,
            "Answer": answer,
            "Type": "Paraphrase"
        })

# Create a new DataFrame
new_df = pd.DataFrame(new_rows)

# Save to a new Excel file
new_df.to_excel("rephrased_questions.xlsx", index=False)
print("Done! Rephrased questions saved to rephrased_questions.xlsx")
