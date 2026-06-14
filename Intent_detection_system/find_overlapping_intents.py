"""
Module to identify overlapping intents between domains using embedding similarity.
"""
import os
import json
import numpy as np
from typing import List, Tuple

from VamsiSir import embeddingVars

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
INTENT_CENTROIDS_PATH = os.path.join(ARTIFACTS_DIR, "intent_centroids.json")


def load_intent_centroids() -> dict:
    if not os.path.exists(INTENT_CENTROIDS_PATH):
        raise FileNotFoundError(f"{INTENT_CENTROIDS_PATH} not found. Run centroid generation first.")
    with open(INTENT_CENTROIDS_PATH, "r") as f:
        return json.load(f)


def find_overlapping_intents(intent_centroids: dict, domain_registry: dict, threshold: float = 0.85) -> List[Tuple[str, str, str, str, float]]:
    overlaps = []
    for domain_a, info_a in domain_registry.items():
        for intent_a in info_a['intents']:
            if intent_a not in intent_centroids:
                continue
            vec_a = np.array(intent_centroids[intent_a])
            for domain_b, info_b in domain_registry.items():
                if domain_a == domain_b:
                    continue
                for intent_b in info_b['intents']:
                    if intent_b not in intent_centroids:
                        continue
                    vec_b = np.array(intent_centroids[intent_b])
                    sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
                    if sim > threshold:
                        overlaps.append((intent_a, domain_a, intent_b, domain_b, sim))
    return overlaps


def print_overlaps(overlaps: List[Tuple[str, str, str, str, float]]):
    if not overlaps:
        print("No overlapping intents found above the threshold.")
        return
    print("Overlapping intents (similarity > threshold):")
    for a, da, b, db, sim in overlaps:
        print(f"{a} ({da}) ↔ {b} ({db}): {sim:.2f}")


def main():
    intent_centroids = load_intent_centroids()
    domain_registry = embeddingVars.DOMAIN_REGISTRY
    threshold = 0.85  # You can adjust this threshold
    overlaps = find_overlapping_intents(intent_centroids, domain_registry, threshold)
    print_overlaps(overlaps)


if __name__ == "__main__":
    main()
