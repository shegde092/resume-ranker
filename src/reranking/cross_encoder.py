import logging
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Any

from src.config import settings


logger = logging.getLogger(__name__)

class ONNXCrossEncoder:
    def __init__(self, model_path: str = "models/ms_marco_minilm_l6_v2_int8.onnx", model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
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
        
        total_tokenization_time = 0.0
        total_inference_time = 0.0
        total_postprocessing_time = 0.0
        
        # Process candidates in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            try:
                # 1. Tokenize pairs
                t_token_start = time.time()
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
                
                t_token_duration = time.time() - t_token_start
                total_tokenization_time += t_token_duration
                    
                # 2. ONNX Inference
                t_infer_start = time.time()
                logits = self.session.run(None, ort_inputs)[0]
                t_infer_duration = time.time() - t_infer_start
                total_inference_time += t_infer_duration
                
                # 3. Postprocess logits
                t_post_start = time.time()
                logits = np.clip(logits, -50, 50)
                if logits.ndim > 2:
                    raise ValueError(f"Expected 1D or 2D logits from cross-encoder, got {logits.ndim}D tensor.")
                elif logits.ndim > 1 and logits.shape[1] == 1:
                    scores = 1 / (1 + np.exp(-logits.flatten()))
                elif logits.ndim > 1:
                    scores = 1 / (1 + np.exp(-logits[:, 1]))
                else:
                    scores = 1 / (1 + np.exp(-logits))
                all_scores.extend(scores.tolist())
                t_post_duration = time.time() - t_post_start
                total_postprocessing_time += t_post_duration
                
            except Exception as e:
                logger.error(f"CE batch inference failed: {e}")
                # Return 0.0 for this batch so a single bad candidate doesn't crash the entire request
                all_scores.extend([0.0] * len(batch_pairs))
                
        logger.info(f"CE Profiling: Tokenization={total_tokenization_time:.4f}s | ONNX Inference={total_inference_time:.4f}s | Postprocess={total_postprocessing_time:.4f}s")
        return all_scores

    def _safe_scalar(self, value):
        import numpy as np

        if isinstance(value, list):
            if len(value) == 0:
                return 0.0
            value = value[0]

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            value = float(np.mean(value))

        return float(value)

    def rerank(self, parsed_jd: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank candidates using matching sectional chunks:
        jd_career <-> cand_career
        jd_skills <-> cand_skills
        jd_profile <-> cand_profile
        jd_education <-> cand_education
        
        Non-empty pairs are predicted in a single unified batch for efficiency.
        """
        if not candidates:
            return []
            
        jd_chunks = parsed_jd.get("chunks", {})
        jd_career = str(jd_chunks.get("career", "")).strip()
        jd_skills = str(jd_chunks.get("skills", "")).strip()
        jd_profile = str(jd_chunks.get("profile", "")).strip()
        jd_education = str(jd_chunks.get("education", "")).strip()
        
        flat_pairs = []
        mapping = {}
        
        sections = [
            ("chunk_career", jd_career, "career"),
            ("chunk_skills", jd_skills, "skills"),
            ("chunk_profile", jd_profile, "profile"),
            ("chunk_education", jd_education, "education")
        ]
        
        for idx, cand in enumerate(candidates):
            for sec_name, jd_text, map_name in sections:
                cand_text = str(cand.get(sec_name, "")).strip()
                if jd_text and cand_text:
                    mapping[(idx, map_name)] = len(flat_pairs)
                    flat_pairs.append((jd_text, cand_text))
                    
        flat_scores = []
        if flat_pairs:
            logger.info(f"Running CE batch prediction on {len(flat_pairs)} non-empty pairs...")
            flat_scores = self.predict(flat_pairs)
            
        w_skills = getattr(settings, 'WEIGHT_CE_SKILLS', 0.40)
        w_career = getattr(settings, 'WEIGHT_CE_CAREER', 0.30)
        w_profile = getattr(settings, 'WEIGHT_CE_PROFILE', 0.20)
        w_education = getattr(settings, 'WEIGHT_CE_EDUCATION', 0.10)
        
        scored_candidates = []
        for idx, cand in enumerate(candidates):
            c_score = flat_scores[mapping[(idx, "career")]] if (idx, "career") in mapping else 0.0
            s_score = flat_scores[mapping[(idx, "skills")]] if (idx, "skills") in mapping else 0.0
            p_score = flat_scores[mapping[(idx, "profile")]] if (idx, "profile") in mapping else 0.0
            e_score = flat_scores[mapping[(idx, "education")]] if (idx, "education") in mapping else 0.0
            
            ce_score = (
                (w_career * c_score) +
                (w_skills * s_score) +
                (w_profile * p_score) +
                (w_education * e_score)
            )
            
            cand_out = cand.copy()
            cand_out.update({
                "ce_career_score": float(c_score),
                "ce_skills_score": float(s_score),
                "ce_profile_score": float(p_score),
                "ce_education_score": float(e_score),
                "ce_score": float(ce_score)
            })
            scored_candidates.append(cand_out)
            
        return scored_candidates
