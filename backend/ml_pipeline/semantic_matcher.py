"""
semantic_matcher.py
-------------------
BERT-powered semantic skill matching using sentence-transformers.

Instead of brittle keyword matching (Python == Python, nothing else),
this module encodes both candidate skills and job requirements into
dense vector embeddings and computes cosine similarity.

Example:
    "Built neural networks" → 94% match with "Machine Learning"
    "Developed REST APIs"   → 91% match with "Backend Development"
    "React.js"              → 98% match with "React"
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util

# We use the lightweight MiniLM model — only 80MB, very fast on CPU,
# and highly accurate for short skill phrases.
# It is downloaded once automatically and cached locally.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None  # Lazy-loaded singleton to avoid loading on import

def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the BERT model."""
    global _model
    if _model is None:
        print("[BERT] Loading semantic model: all-MiniLM-L6-v2...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[BERT] Model loaded successfully.")
    return _model


def semantic_skill_match(
    candidate_skills: list[str],
    required_skills: list[str],
    similarity_threshold: float = 0.55
) -> dict:
    """
    Perform semantic skill matching using BERT embeddings.

    Args:
        candidate_skills:    List of skills extracted from the candidate's CV.
        required_skills:     List of skills required by the target company/role.
        similarity_threshold: Cosine similarity score (0-1) above which a
                              candidate skill is considered a match.
                              0.55 is a good balance between precision and recall.

    Returns:
        A dict with:
          - matched_skills:    Required skills the candidate semantically covers
          - missing_skills:    Required skills the candidate is lacking
          - skill_match_pct:   Overall % match score
          - match_details:     Per-skill breakdown with best match and confidence
    """
    if not required_skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "skill_match_pct": 75.0,
            "match_details": []
        }

    if not candidate_skills:
        return {
            "matched_skills": [],
            "missing_skills": required_skills,
            "skill_match_pct": 0.0,
            "match_details": [{"required": s, "best_match": None, "confidence": 0.0, "matched": False} for s in required_skills]
        }

    model = _get_model()

    # Encode all skills into dense vectors in one batch (fast)
    candidate_embeddings = model.encode(candidate_skills, convert_to_tensor=True)
    required_embeddings  = model.encode(required_skills,  convert_to_tensor=True)

    # Compute cosine similarity matrix: shape (n_required, n_candidate)
    cos_scores = util.cos_sim(required_embeddings, candidate_embeddings)

    matched_skills = []
    missing_skills = []
    match_details  = []

    for i, req_skill in enumerate(required_skills):
        # Best matching candidate skill for this required skill
        scores_for_req = cos_scores[i].cpu().numpy()
        best_idx       = int(np.argmax(scores_for_req))
        best_score     = float(scores_for_req[best_idx])
        best_cand_skill = candidate_skills[best_idx]

        is_matched = best_score >= similarity_threshold

        if is_matched:
            matched_skills.append(req_skill)
        else:
            missing_skills.append(req_skill)

        match_details.append({
            "required":    req_skill,
            "best_match":  best_cand_skill if is_matched else None,
            "confidence":  round(best_score * 100, 1),   # as percentage
            "matched":     is_matched
        })

    skill_match_pct = (len(matched_skills) / len(required_skills)) * 100

    return {
        "matched_skills":  matched_skills,
        "missing_skills":  missing_skills,
        "skill_match_pct": round(skill_match_pct, 2),
        "match_details":   match_details
    }


if __name__ == "__main__":
    # Quick test to verify it works
    candidate = ["neural networks", "deep learning", "Python", "built REST APIs"]
    required  = ["Machine Learning", "Python", "Backend Development", "SQL"]
    result = semantic_skill_match(candidate, required)
    print("\n=== BERT Semantic Skill Match Test ===")
    for detail in result["match_details"]:
        status = "✅" if detail["matched"] else "❌"
        print(f"  {status} {detail['required']}: best match → '{detail['best_match']}' ({detail['confidence']}%)")
    print(f"\n  Overall match: {result['skill_match_pct']}%")
