"""Prevent interrupted and partial histories entering the finished-run archive."""
import unittest
from tools.organize_completed_research import classify


class CompletionTests(unittest.TestCase):
    def test_full_history_is_required_even_with_completed_status(self):
        self.assertFalse(classify({'args': {'epochs': 3}},
                                  {'status': 'completed'}, [{'epoch': '3'}])[0])

    def test_running_and_interrupted_are_rejected_even_with_full_history(self):
        for state in ('running', 'interrupted', 'failed', 'dry_run_completed'):
            self.assertFalse(classify({'args': {'epochs': 2}}, {'status': state},
                                      [{'epoch': 1}, {'epoch': 2}])[0])

    def test_best_epoch_is_not_used_as_completion_epoch(self):
        self.assertTrue(classify({'args': {'epochs': 3}},
                                 {'status': 'completed', 'best_epoch': 1, 'last_completed_epoch': 3},
                                 [{'epoch': n} for n in (1, 2, 3)])[0])

    def test_status_and_schedule_must_agree(self):
        self.assertFalse(classify({'args': {'epochs': 2}},
                                  {'status': 'completed', 'last_completed_epoch': 1},
                                  [{'epoch': 1}, {'epoch': 2}])[0])

    def test_unknown_plan_and_dry_run_are_rejected(self):
        self.assertFalse(classify({}, {}, [{'epoch': 1}])[0])
        self.assertFalse(classify({'args': {'epochs': 1, 'dry_run': True}}, {}, [{'epoch': 1}])[0])


if __name__ == '__main__':
    unittest.main()
