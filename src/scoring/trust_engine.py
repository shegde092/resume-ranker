import logging
from typing import Dict, Any, List



logger = logging.getLogger(__name__)


class TrustEngine:
    def __init__(self):
        pass

    def evaluate(
        self,
        parsed_resume: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate trust score using soft penalties.
        Returns:
        {
            "trust_score": float,
            "penalties": list
        }
        """
        penalties = []
        base_trust = 1.0

        career_history = parsed_resume.get("career_history", [])
        education = parsed_resume.get("education", [])
        redrob_signals = parsed_resume.get("redrob_signals", {})



        # 2. Suspicious education duration
        education_penalty = self._check_education_duration(education)
        if education_penalty > 0:
            penalties.append("suspicious_degree_duration")
            base_trust -= education_penalty

        # 3. Experience Mismatch Penalty
        exp_mismatch, exp_penalty = self._check_experience_mismatch(parsed_resume.get("years_of_experience"), career_history)
        if exp_penalty > 0:
            penalties.append(f"experience_mismatch_penalty_{int(exp_mismatch)}_months")
            base_trust -= exp_penalty

        # 4. Salary Anomaly
        salary_penalty = self._check_salary_anomaly(parsed_resume.get("salary_metadata"))
        if salary_penalty > 0:
            penalties.append("salary_anomaly_penalty")
            base_trust -= salary_penalty

        # 5. Recruiter behavior signals
        response_rate = redrob_signals.get("recruiter_response_rate", 1.0)
        if response_rate < 0.1:
            penalties.append("low_response_rate")
            base_trust -= 0.1

        trust_score = float(max(0.0, min(1.0, base_trust)))

        return {
            "trust_score": trust_score,
            "penalties": penalties,
            "mismatch_months": exp_mismatch,
            "experience_mismatch_penalty": exp_penalty,
            "salary_anomaly_penalty": salary_penalty
        }



    def _check_education_duration(
        self,
        education: List[Dict[str, Any]]
    ) -> float:
        """
        Soft penalty for suspicious degree duration.
        """
        penalty = 0.0

        for edu in education:
            start = edu.get("start_year")
            end = edu.get("end_year")
            degree = str(edu.get("degree", "")).lower()

            if start is None or end is None:
                continue

            try:
                duration = int(end) - int(start)

                if "phd" in degree and duration < 2:
                    penalty += 0.15

                elif ("b.e" in degree or "btech" in degree or "bachelor" in degree) and duration < 3:
                    penalty += 0.1

            except (TypeError, ValueError):
                continue

        return penalty

    def _check_experience_mismatch(self, stated_yoe, career_history: List[Dict[str, Any]]):
        """
        Compare stated years of experience against actual summed career duration.
        Returns: (mismatch_months: float, penalty: float)
        """
        if stated_yoe is None or not career_history:
            return 0.0, 0.0
            
        try:
            stated_yoe = float(stated_yoe)
        except (ValueError, TypeError):
            return 0.0, 0.0
            
        total_years = 0.0
        from datetime import datetime
        current_year = datetime.now().year
        
        for exp in career_history:
            start = exp.get("start_year")
            end = exp.get("end_year")
            
            if start is None:
                continue
                
            try:
                start = float(start)
                if end is None or (isinstance(end, str) and end.lower() in ["present", "current", "now"]):
                    end = current_year
                else:
                    end = float(end)
                    
                if end >= start:
                    total_years += (end - start)
            except (ValueError, TypeError):
                continue
                
        diff_years = stated_yoe - total_years
        
        # Only penalize if stated YOE is significantly HIGHER than computed total
        if diff_years > 2.0:
            mismatch_months = diff_years * 12.0
            # Scale penalty: 0.1 for every extra year over 2
            penalty = min(0.3, (diff_years - 2.0) * 0.1)
            return mismatch_months, penalty
            
        return 0.0, 0.0

    def _check_salary_anomaly(self, salary_metadata: Dict[str, Any]) -> float:
        """
        Check for min > max or malformed salary metadata if present.
        Returns penalty: float
        """
        if not salary_metadata:
            return 0.0
            
        penalty = 0.0
        
        min_salary = salary_metadata.get("min_salary")
        max_salary = salary_metadata.get("max_salary")
        
        if min_salary is not None and max_salary is not None:
            try:
                min_s = float(min_salary)
                max_s = float(max_salary)
                if min_s > max_s:
                    penalty += 0.2
                if min_s < 0 or max_s < 0:
                    penalty += 0.2
            except (ValueError, TypeError):
                penalty += 0.1
                
        return min(0.4, penalty)


