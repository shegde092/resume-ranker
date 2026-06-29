import logging
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Any

from src.config import settings


logger = logging.getLogger(__name__)

class ONNXCrossEncoder:
    def __init__(self, model_path: str = "models/minilm_l6_v2_int8.onnx", model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize ONNX INT8 cross encoder for fast CPU inference.
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create session with optimizations tailored for CPU latency budget
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Force single/dual thread per session for throughput consistency in concurrent requests
        sess_options.intra_op_num_threads = 2
        # Improve ONNX CPU thread scheduling stability
        sess_options.inter_op_num_threads = 1
        
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=["CPUExecutionProvider"])
            self.is_loaded = True
            logger.info(f"Loaded ONNX CrossEncoder from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model at {model_path}: {e}")
    def warmup(self):
        """
        Run one dummy inference to reduce first-request latency.
        """
        if self.is_loaded:
            dummy_pairs = [("warmup query", "warmup document")]
            _ = self.predict(dummy_pairs, batch_size=1)
            logger.info("ONNX CrossEncoder warmup complete.")
            
    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = None) -> List[float]:
        """
        Score a list of (query, document) pairs using batched inference.
        """
        if not pairs:
            return []
            
        if batch_size is None:
            batch_size = getattr(settings, 'CE_BATCH_SIZE', 32)
            
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
            
        if not self.is_loaded:
            raise RuntimeError("ONNX model is not loaded. Cannot perform inference.")
            
        all_scores = []
        
        # Process candidates in batches
        for i in range(0, len(pairs), batch_size):
            batch_start_time = time.time()
            batch_pairs = pairs[i:i + batch_size]
            
            try:
                # Tokenize pairs
                inputs = self.tokenizer(
                    batch_pairs, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="np"
                )
                
                ort_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }
                
                if "token_type_ids" in inputs and any(i_def.name == "token_type_ids" for i_def in self.session.get_inputs()):
                    ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
                    
                # Inference
                logits = self.session.run(None, ort_inputs)[0]
                
                # Prevent sigmoid numerical overflow
                logits = np.clip(logits, -50, 50)
                
                # Convert logits to probabilities
                if logits.ndim > 1 and logits.shape[1] == 1:
                    scores = 1 / (1 + np.exp(-logits.flatten()))
                elif logits.ndim > 1:
                    scores = 1 / (1 + np.exp(-logits[:, 1]))
                else:
                    scores = 1 / (1 + np.exp(-logits))
                    
                all_scores.extend(scores.tolist())

            except Exception as e:
                logger.error(f"CE batch inference failed: {e}")
                # Return 0.0 for this batch so a single bad candidate doesn't crash the entire request
                all_scores.extend([0.0] * len(batch_pairs))
                
            batch_duration = time.time() - batch_start_time
            logger.info(f"CE Batch latency: {batch_duration:.3f}s")
            
        return all_scores

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 500) -> List[Dict[str, Any]]:
        """
        Rerank retrieved candidates using 4 separate semantic section scores.
        Ensures strict latency budget compliance.
        """
        start_time = time.time()
        budget = settings.BUDGET_CROSS_ENCODER
        
        if not candidates:
            return []
            
        # Lightweight pre-filter step (avoid in-place mutation of caller's list)
        if all("score" in cand for cand in candidates):
            candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
            
        ce_limit = getattr(settings, 'MAX_CANDIDATES_CE', 800)
        ce_candidates = candidates[:ce_limit]
        num_cands = len(ce_candidates)
            
        career_pairs = [(str(query or ""), str(cand.get("career_text") or "")) for cand in ce_candidates]
        skill_pairs = [(str(query or ""), str(cand.get("skills_text") or "")) for cand in ce_candidates]
        profile_pairs = [(str(query or ""), str(cand.get("profile_text") or "")) for cand in ce_candidates]
        edu_pairs = [(str(query or ""), str(cand.get("education_text") or "")) for cand in ce_candidates]
        
        # 1. Career
        career_scores = self.predict(career_pairs)
        
        # 2. Skills
        elapsed = time.time() - start_time
        if elapsed > budget * 0.5:
            fallback = getattr(settings, 'FALLBACK_CE_CANDIDATES', 500)
            reduced_limit = min(num_cands, top_k, fallback)
            skill_pairs = skill_pairs[:reduced_limit]
            profile_pairs = profile_pairs[:reduced_limit]
            edu_pairs = edu_pairs[:reduced_limit]
        else:
            reduced_limit = num_cands
            
        skill_scores_raw = self.predict(skill_pairs)
        skill_scores = skill_scores_raw + [0.0] * (num_cands - len(skill_scores_raw))
        
        # Track which components were skipped to renormalize weights
        skip_profile = False
        skip_edu = False

        # Adaptive degradation based on candidate volume
        if num_cands > 700:
            logger.warning("Adaptive degradation: skipping profile and education scoring (cands > 700)")
            profile_scores = [0.0] * num_cands
            edu_scores = [0.0] * num_cands
            skip_profile = True
            skip_edu = True
        else:
            # 3. Profile
            elapsed = time.time() - start_time
            if elapsed > budget * 0.75:
                logger.warning("CE latency budget critical, skipping profile scoring")
                profile_scores = [0.0] * num_cands
                skip_profile = True
                
                # Apply secondary degradation for remaining steps
                min_cands = getattr(settings, 'MIN_CE_CANDIDATES', 300)
                reduced_limit = min(reduced_limit, min_cands)
                edu_pairs = edu_pairs[:reduced_limit]
            else:
                profile_scores_raw = self.predict(profile_pairs)
                profile_scores = profile_scores_raw + [0.0] * (num_cands - len(profile_scores_raw))
                
            if num_cands > 600:
                logger.warning("Adaptive degradation: skipping education scoring (cands > 600)")
                edu_scores = [0.0] * num_cands
                skip_edu = True
            else:
                # 4. Education
                elapsed = time.time() - start_time
                if elapsed > budget * 0.90:
                    logger.warning("CE latency budget exhausted, skipping education scoring")
                    edu_scores = [0.0] * num_cands
                    skip_edu = True
                else:
                    edu_scores_raw = self.predict(edu_pairs)
                    edu_scores = edu_scores_raw + [0.0] * (num_cands - len(edu_scores_raw))
        
        scored_candidates = []
        for i, cand in enumerate(ce_candidates):
            # Safe score indexing
            c_fit = float(career_scores[i]) if i < len(career_scores) else 0.0
            s_fit = float(skill_scores[i]) if i < len(skill_scores) else 0.0
            p_fit = float(profile_scores[i]) if i < len(profile_scores) else 0.0
            e_fit = float(edu_scores[i]) if i < len(edu_scores) else 0.0
            
            # Preserve existing candidate fields
            cand_out = cand.copy()
            cand_out.update({
                "career_fit_ce": c_fit,
                "skill_fit_ce": s_fit,
                "profile_fit_ce": p_fit,
                "education_fit_ce": e_fit
            })
            
            # Adaptive Weight Normalization
            weight_sum = 0.4 + 0.3
            score_sum = (0.4 * c_fit) + (0.3 * s_fit)
            
            if not skip_profile:
                weight_sum += 0.2
                score_sum += (0.2 * p_fit)
                
            if not skip_edu:
                weight_sum += 0.1
                score_sum += (0.1 * e_fit)
            
            ce_score_avg = score_sum / weight_sum if weight_sum > 0 else 0.0
            cand_out["ce_score_avg"] = ce_score_avg
            
            scored_candidates.append((ce_score_avg, cand_out))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        final_candidates = [cand for _, cand in scored_candidates[:top_k]]
        
        total_duration = time.time() - start_time
        logger.info(f"CrossEncoder reranking took {total_duration:.3f}s")
        
        return final_candidates
