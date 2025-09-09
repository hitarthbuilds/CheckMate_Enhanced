# ml_model_multilingual/evidence.py

def fetch_evidence(claim: str) -> str:
    """
    Placeholder evidence retrieval. Replace with real search or knowledge base.
    """
    knowledge_base = {
        "The Earth is flat.": "The Earth is round and spherical, as confirmed by astronomy and satellite images.",
        "Vaccines cause autism.": "Scientific studies show no link between vaccines and autism.",
        "Climate change is a hoax.": "Scientific consensus confirms climate change is real and caused by humans."
    }
    return knowledge_base.get(claim, "Evidence not found. Use a proper search API.")
