from typing import Dict, Any

class TemplateBuilder:
    @staticmethod
    def build_candidate_chunks(candidate_dict: Dict[str, Any]) -> Dict[str, str]:
        profile = candidate_dict.get("profile", {})
        redrob_signals = candidate_dict.get("redrob_signals", {})
        
        # 1. Career
        yoe = candidate_dict.get("years_of_experience") or profile.get("years_of_experience", "N/A")
        career_parts = [f"Stated Years of Experience: {yoe}"]
        for exp in candidate_dict.get("career_history", []):
            title = exp.get("title", "N/A")
            company = exp.get("company", "N/A")
            dur = exp.get("duration_months", "N/A")
            desc = exp.get("description", "")
            career_parts.append(f"Role: {title} at {company} for {dur} months. Description: {desc}")
        career_text = " | ".join(career_parts)
        
        # 2. Skills
        skills = candidate_dict.get("skills", [])
        skills_list = []
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, dict):
                    name = s.get("name") or s.get("skill_name") or ""
                    prof = s.get("proficiency", "")
                    dur = s.get("duration_months", 0)
                    skills_list.append(f"{name} ({prof}, {dur}m)")
                else:
                    skills_list.append(str(s))
        skills_str = ", ".join(skills_list)
        
        certs = candidate_dict.get("certifications", [])
        certs_list = []
        for c in certs:
            if isinstance(c, dict):
                certs_list.append(f"{c.get('name', '')} by {c.get('issuer', '')}")
            else:
                certs_list.append(str(c))
        certs_str = ", ".join(certs_list)
        skills_text = f"Skills: {skills_str}. Certifications: {certs_str}."
        
        # 3. Profile
        loc = candidate_dict.get("location") or profile.get("location", "N/A")
        country = profile.get("country", "N/A")
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        
        salary_expect = candidate_dict.get("salary_expectation") or redrob_signals.get("expected_salary_range_inr_lpa", {})
        if isinstance(salary_expect, dict):
            salary_str = f"{salary_expect.get('min', 'N/A')}-{salary_expect.get('max', 'N/A')} LPA"
        else:
            salary_str = str(salary_expect)
            
        notice = candidate_dict.get("notice_period") or redrob_signals.get("notice_period_days", "N/A")
        relocate = "Yes" if candidate_dict.get("open_to_relocate") or redrob_signals.get("willing_to_relocate") else "No"
        open_to_work = "Yes" if candidate_dict.get("open_to_work") or redrob_signals.get("open_to_work_flag") else "No"
        work_mode = redrob_signals.get("preferred_work_mode", "N/A")
        
        response_rate = redrob_signals.get("recruiter_response_rate", "N/A")
        github_score = redrob_signals.get("github_activity_score", "N/A")
        profile_completeness = redrob_signals.get("profile_completeness_score", "N/A")
        
        profile_text = (
            f"Headline: {headline} | Summary: {summary} | "
            f"Location: {loc}, {country} | Willing to Relocate: {relocate} | Preferred Work Mode: {work_mode} | "
            f"Salary Expectation: {salary_str} | Notice Period: {notice} days | Open to Work: {open_to_work} | "
            f"Profile Completeness: {profile_completeness}% | Recruiter Response Rate: {response_rate} | GitHub Activity: {github_score}"
        )
        
        # 4. Education
        edu_parts = []
        for edu in candidate_dict.get("education", []):
            degree = edu.get("degree", "N/A")
            field = edu.get("field_of_study", "N/A")
            inst = edu.get("institution") or edu.get("school", "N/A")
            start = edu.get("start_year", "N/A")
            end = edu.get("end_year", "N/A")
            grade = edu.get("grade", "N/A")
            tier = edu.get("tier", "N/A")
            edu_parts.append(f"{degree} in {field} from {inst} ({start}-{end}), Grade: {grade}, Tier: {tier}")
        education_text = " | ".join(edu_parts)
        
        return {
            "career": career_text,
            "skills": skills_text,
            "profile": profile_text,
            "education": education_text
        }
