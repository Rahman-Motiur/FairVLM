import torch
import random
from sentence_transformers import SentenceTransformer, util


class SRCP:
    """
    Semantic-Retaining Counterfactual Prompting:
    Generates m prompt variations and selects top k based on:
    - Lexical diversity (Jaccard ~ 0.3–0.5)
    - Semantic similarity (cosine >= 0.90)
    """

    def __init__(self, m=5, k=3):
        self.m = m
        self.k = k
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def jaccard(self, a, b):
        A = set(a.lower().split())
        B = set(b.lower().split())
        return len(A & B) / len(A | B)

    def generate_single(self, prompt):
        """
        Placeholder synthetic variation. Replace with GPT-4o API if available.
        """
        variations = [
            prompt.replace("the", "a"),
            prompt + " region",
            "segment the " + prompt,
            prompt + " boundary",
            prompt + " area"
        ]
        return random.choice(variations)

    def generate(self, prompts):
        cf_prompts = []

        for p in prompts:
            # generate m candidates
            candidates = [self.generate_single(p) for _ in range(self.m)]

            # encode prompts
            embeds = self.encoder.encode([p] + candidates, convert_to_tensor=True)
            p_emb = embeds[0]
            cand_emb = embeds[1:]

            scored = []
            for c, emb in zip(candidates, cand_emb):
                sim = util.cos_sim(p_emb, emb).item()
                jac = self.jaccard(p, c)
                scored.append((c, sim, jac))

            # filter valid counterfactual prompts
            valid = [x for x in scored if x[1] >= 0.90 and 0.3 <= x[2] <= 0.5]

            # sort by similarity
            valid = sorted(valid, key=lambda x: x[1], reverse=True)

            # take top k
            chosen = [v[0] for v in valid[:self.k]]

            cf_prompts.append(chosen if chosen else [p])

        return cf_prompts
