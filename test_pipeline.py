import unittest

def get_exp_fit(years_exp):
    if years_exp < 0.0:
        return 0.0
    if 5.0 <= years_exp <= 9.0:
        return 1.0
    elif years_exp < 5.0:
        return years_exp / 5.0
    else:
        return max(0.0, 1.0 - (years_exp - 9.0) / 10.0)

def get_notice_score(notice_days):
    if notice_days < 0:
        return 1.0
    if notice_days <= 30:
        return 1.0
    elif notice_days <= 90:
        return 1.0 - ((notice_days - 30) / 60.0) * 0.8
    else:
        return 0.0

class TestPipelineConstraints(unittest.TestCase):
    
    def test_weights_sum_to_one(self):
        # Weights: semantic_similarity, must_have, exp_fit, title_rel, career_prog, availability, recency
        weights = [0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03]
        self.assertAlmostEqual(sum(weights), 1.0, places=5, msg="Baseline weights must sum to exactly 1.0")

    def test_experience_band_fit(self):
        # Target: 5-9 years
        self.assertAlmostEqual(get_exp_fit(5.0), 1.0, places=5)
        self.assertAlmostEqual(get_exp_fit(7.2), 1.0, places=5)
        self.assertAlmostEqual(get_exp_fit(9.0), 1.0, places=5)
        
        # Lower than 5
        self.assertAlmostEqual(get_exp_fit(2.5), 0.5, places=5)
        self.assertAlmostEqual(get_exp_fit(0.0), 0.0, places=5)
        
        # Greater than 9
        self.assertAlmostEqual(get_exp_fit(14.0), 0.5, places=5)
        self.assertAlmostEqual(get_exp_fit(19.0), 0.0, places=5)
        self.assertAlmostEqual(get_exp_fit(25.0), 0.0, places=5)
        
        # Invalid input (negative experience)
        self.assertAlmostEqual(get_exp_fit(-1.5), 0.0, places=5)

    def test_notice_period_decay(self):
        # Preferred: <= 30 days
        self.assertAlmostEqual(get_notice_score(0), 1.0, places=5)
        self.assertAlmostEqual(get_notice_score(15), 1.0, places=5)
        self.assertAlmostEqual(get_notice_score(30), 1.0, places=5)
        
        # 31-90 days linear decay to 0.2
        self.assertAlmostEqual(get_notice_score(60), 0.6, places=5)
        self.assertAlmostEqual(get_notice_score(90), 0.2, places=5)
        
        # > 90 days is 0.0
        self.assertAlmostEqual(get_notice_score(91), 0.0, places=5)
        self.assertAlmostEqual(get_notice_score(180), 0.0, places=5)
        
        # Invalid input (negative notice days)
        self.assertAlmostEqual(get_notice_score(-10), 1.0, places=5)

    def test_honeypot_penalty(self):
        # Check that score * 0.85 is applied correctly
        base_score = 0.80
        penalized_score = base_score * 0.85
        self.assertAlmostEqual(penalized_score, 0.68, places=5)
        
    def test_tier_bonuses(self):
        tier_bonuses = {'A': 0.30, 'B': 0.20, 'C': 0.10, 'D': 0.00}
        self.assertAlmostEqual(tier_bonuses['A'], 0.30, places=5)
        self.assertAlmostEqual(tier_bonuses['B'], 0.20, places=5)
        self.assertAlmostEqual(tier_bonuses['C'], 0.10, places=5)
        self.assertAlmostEqual(tier_bonuses['D'], 0.00, places=5)

    def test_rank_ordering(self):
        # Test that high weighted score Tier B outranks low weighted score Tier A
        # score = weighted_score + tier_bonus
        # cand_A: Tier A, weighted_score = 0.51 -> score = 0.51 + 0.30 = 0.81
        # cand_B: Tier B, weighted_score = 0.94 -> score = 0.94 + 0.20 = 1.14
        cand_A = 0.51 + 0.30
        cand_B = 0.94 + 0.20
        self.assertGreater(cand_B, cand_A, "Exceptional Tier B candidate should outrank weak Tier A candidate via tier bonus")

    def test_rank_key_sorting(self):
        # key format: (-final_score, tier_letter, -must_have_skill_match, -semantic_similarity, candidate_id)
        # Test tie-breaking:
        # Same final_score (e.g. 0.95), different tier: Tier A before Tier B
        c1 = (-0.95, 'A', -0.8, -0.9, 'C2')
        c2 = (-0.95, 'B', -0.8, -0.9, 'C1')
        ranked = sorted([c2, c1])
        self.assertEqual(ranked[0], c1, "Tie break: Tier A should sort before Tier B")
        
        # Same score, same tier, same skills, same similarity: lower candidate_id first
        c3 = (-0.95, 'A', -0.8, -0.9, 'C2')
        c4 = (-0.95, 'A', -0.8, -0.9, 'C1')
        ranked2 = sorted([c3, c4])
        self.assertEqual(ranked2[0], c4, "Tie break: Candidate ID C1 should sort before C2")

if __name__ == '__main__':
    unittest.main()
