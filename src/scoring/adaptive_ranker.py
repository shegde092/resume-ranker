import logging
import time
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

from src.config import settings
from src.scoring.feature_assembler import FeatureAssembler

logger = logging.getLogger(__name__)

class AdaptiveFusionRanker:
    """
    Computes final candidate score using a hybrid multiplicative-additive fusion strategy:
    Final = SemanticScore * (0.3 + 0.7 * Availability * Location * Trust * AntiPersona)
    Uses all meaningful candidate schema signals and behavioral metrics.
    """
    def __init__(self):
        self.feature_assembler = FeatureAssembler()

    def rank(self, candidates: List[Dict[str, Any]], feature_matrix: np.ndarray = None, parsed_jd: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Executes the hybrid score fusion leveraging all schema variables.
        """
        if not candidates:
            return []
            
        start_time = time.time()
        
        if parsed_jd is None and len(candidates) > 0:
            from src.parsing.jd_parser import JDParser
            parser = JDParser()
            parsed_jd = parser.parse(candidates[0].get("query", ""))

        scored_candidates = []
        for cand in candidates:
            cand_out = cand.copy()
            
            # 1. Semantic Score (Weighted Cross Encoder score)
            semantic_score = float(cand.get("ce_score", 0.0))
            
            # 2. Availability Multiplier
            availability_mult = self._compute_availability(cand)
            
            # 3. Location Multiplier
            location_mult = self._compute_location(cand, parsed_jd) if parsed_jd else 1.0
            
            # 4. Trust Penalty
            trust_score = float(cand.get("trust_score", 1.0))
            is_suspicious = cand.get("is_suspicious", False)
            honeypot_score = float(cand.get("honeypot_score", 1.0))
            
            combined_trust = trust_score
            if is_suspicious or honeypot_score < 0.5:
                combined_trust *= 0.1
                
            trust_penalty = max(0.05, combined_trust)
            
            # 5. Anti-Persona Penalty
            anti_persona_penalty = self._compute_anti_persona(cand, parsed_jd) if parsed_jd else 1.0
            
            # Strictly Multiplicative Fusion
            final_score = (
                semantic_score *
                availability_mult *
                location_mult *
                trust_penalty *
                anti_persona_penalty
            )
            
            cand_out.update({
                "semantic_score": semantic_score,
                "availability_multiplier": availability_mult,
                "location_multiplier": location_mult,
                "trust_penalty": trust_penalty,
                "anti_persona_penalty": anti_persona_penalty,
                "final_score": float(max(0.0, min(1.0, final_score)))
            })
            scored_candidates.append(cand_out)
            
        scored_candidates.sort(key=lambda x: (-float(x.get("final_score", 0.0)), str(x.get("candidate_id", ""))))
        
        total_duration = time.time() - start_time
        logger.info(f"Multiplicative fusion took {total_duration:.3f}s for {len(candidates)} candidates")
        
        return scored_candidates

    def _compute_availability(self, cand: Dict[str, Any]) -> float:
        redrob_signals = cand.get("redrob_signals", {})
        
        # open_to_work
        otw = cand.get("open_to_work_flag") or cand.get("open_to_work")
        if otw is None:
            otw = redrob_signals.get("open_to_work_flag", 1)
        otw_factor = 1.0 if (otw == 1 or otw == "Yes" or otw is True) else 0.15
        
        # recruiter response rate
        rr = cand.get("recruiter_response_rate")
        if rr is None:
            rr = redrob_signals.get("recruiter_response_rate", 1.0)
        try:
            rr_val = float(rr)
        except:
            rr_val = 1.0
        rr_factor = 0.8 + 0.2 * rr_val
        
        # notice period
        notice = cand.get("notice_period_days") or cand.get("notice_period")
        if notice is None:
            notice = redrob_signals.get("notice_period_days", 30)
        try:
            notice_days = int(notice)
        except:
            notice_days = 30
        if notice_days <= 15:
            notice_factor = 1.0
        elif notice_days <= 30:
            notice_factor = 0.9
        elif notice_days <= 60:
            notice_factor = 0.7
        else:
            notice_factor = 0.5
            
        # last active date (compute days delta from YYYY-MM-DD strings or float/ints)
        last_active = cand.get("last_active_date") or redrob_signals.get("last_active_date")
        active_days = 30
        if last_active is not None:
            try:
                active_days = int(last_active)
            except (ValueError, TypeError):
                try:
                    dt = datetime.strptime(str(last_active).strip()[:10], "%Y-%m-%d")
                    active_days = abs((datetime.now() - dt).days)
                except Exception:
                    active_days = 30
                    
        if active_days <= 30:
            active_factor = 1.0
        elif active_days <= 90:
            active_factor = 0.9
        elif active_days <= 180:
            active_factor = 0.7
        else:
            active_factor = 0.5

        # profile completeness
        completeness = redrob_signals.get("profile_completeness_score", 100)
        try:
            comp_val = float(completeness) / 100.0
        except:
            comp_val = 1.0
        comp_factor = 0.9 + 0.1 * comp_val
        
        # interview completion
        ic = redrob_signals.get("interview_completion_rate", 1.0)
        try:
            ic_val = float(ic)
        except:
            ic_val = 1.0
        ic_factor = 0.9 + 0.1 * ic_val
        
        # github activity
        gh = redrob_signals.get("github_activity_score", 0)
        try:
            gh_val = float(gh) / 100.0
        except:
            gh_val = 0.0
        gh_factor = 1.0 + 0.05 * gh_val
        
        # offer acceptance rate
        oa = redrob_signals.get("offer_acceptance_rate", 1.0)
        try:
            oa_val = float(oa)
        except:
            oa_val = 1.0
        oa_factor = 0.9 + 0.1 * oa_val
            
        mult = otw_factor * rr_factor * notice_factor * active_factor * comp_factor * ic_factor * gh_factor * oa_factor
        # Geometrically smoothed to prevent behavioral metrics from over-crushing semantic scores
        smoothed_mult = mult ** 0.5
        return max(0.15, smoothed_mult)

    def _compute_location(self, cand: Dict[str, Any], parsed_jd: Dict[str, Any]) -> float:
        jd_locs = [l.lower() for l in parsed_jd.get("locations", [])]
        jd_mode = str(parsed_jd.get("work_mode", "")).lower()
        
        profile = cand.get("profile", {})
        cand_loc = str(cand.get("location") or profile.get("location", "")).lower()
        
        redrob_signals = cand.get("redrob_signals", {})
        relocate = cand.get("open_to_relocate") or redrob_signals.get("willing_to_relocate")
        
        city_match = False
        if jd_locs:
            city_match = any(loc in cand_loc for loc in jd_locs)
            
        if city_match or "remote" in jd_locs or jd_mode == "remote":
            loc_factor = 1.0
        elif relocate == 1 or relocate == "Yes" or relocate is True:
            loc_factor = 0.7
        else:
            loc_factor = 0.20
            
        cand_mode = str(redrob_signals.get("preferred_work_mode", "")).lower()
        mode_factor = 1.0
        if jd_mode in ["hybrid", "onsite"] and "remote" in cand_mode and not city_match:
            mode_factor = 0.3
            
        # Non-biased dynamic country match parsed from JD
        jd_country = parsed_jd.get("country")
        country_factor = 1.0
        if jd_country:
            jd_country_lower = jd_country.lower()
            cand_country = str(profile.get("country", "")).lower()
            if cand_country and cand_country != "n/a" and cand_country != jd_country_lower and jd_country_lower not in cand_loc:
                country_factor = 0.8
            
        return max(0.20, loc_factor * mode_factor * country_factor)

    def _compute_anti_persona(self, cand: Dict[str, Any], parsed_jd: Dict[str, Any]) -> float:
        jd_anti = parsed_jd.get("anti_personas", {})
        
        jd_severity_consulting = jd_anti.get("consulting", 0.0)
        cand_consulting_ratio = cand.get("consulting_ratio", 0.0)
        consulting_penalty = 1.0
        if jd_severity_consulting > 0:
            consulting_penalty = 1.0 - (jd_severity_consulting * cand_consulting_ratio)
            
        # Refactored research anti-persona: uses structured JSON fields instead of raw text serialization
        jd_severity_research = jd_anti.get("research_only", 0.0)
        is_research = 0.0
        research_keywords = ["postdoc", "academic", "publications", "researcher", "research only", "research fellow", "phd"]
        
        for edu in cand.get("education", []):
            field = str(edu.get("field_of_study", "")).lower()
            degree = str(edu.get("degree", "")).lower()
            if any(w in field or w in degree for w in research_keywords):
                is_research = 1.0
                break
                
        if is_research == 0.0:
            for exp in cand.get("career_history", []):
                title = str(exp.get("title", "")).lower()
                desc = str(exp.get("description", "")).lower()
                if any(w in title or w in desc for w in research_keywords):
                    is_research = 1.0
                    break
                    
        research_penalty = 1.0
        if jd_severity_research > 0:
            research_penalty = 1.0 - (jd_severity_research * is_research)
            
        jd_severity_wrapper = jd_anti.get("framework_wrapper", 0.0)
        skills_str = str(cand.get("chunk_skills", cand.get("skills", ""))).lower()
        career_str = str(cand.get("chunk_career", cand.get("career", ""))).lower()
        is_wrapper = 1.0 if any(w in skills_str or w in career_str for w in ["langchain", "wrapper", "tutorial"]) else 0.0
        wrapper_penalty = 1.0
        if jd_severity_wrapper > 0:
            wrapper_penalty = 1.0 - (jd_severity_wrapper * is_wrapper)
            
        jd_severity_title = jd_anti.get("title_chaser", 0.0)
        cand_title_chaser = cand.get("title_chaser_score", 0.0)
        title_penalty = 1.0
        if jd_severity_title > 0:
            title_penalty = 1.0 - (jd_severity_title * cand_title_chaser)
            
        penalty = consulting_penalty * research_penalty * wrapper_penalty * title_penalty
        return max(0.05, penalty)
