import unittest
from evaluate_false_alarms import clip_bounds, merge_alerts, negative_summary
import gunshot as g


class FalseAlarmTests(unittest.TestCase):
    def test_millisecond_tail_is_not_an_extra_test_example(self):
        bounds = clip_bounds(300 * g.SR + 96)
        self.assertEqual(len(bounds), 30)
        self.assertEqual(bounds[-1], [290 * g.SR, 300 * g.SR + 96])
        self.assertEqual(sum(end - start for start, end in bounds), 300 * g.SR + 96)

    def test_valid_three_second_tail_and_whole_short_source_remain(self):
        self.assertEqual(clip_bounds(13 * g.SR), [[0, 10 * g.SR], [10 * g.SR, 13 * g.SR]])
        self.assertEqual(clip_bounds(g.SR), [[0, g.SR]])

    def test_continuous_flags_form_one_alert(self):
        self.assertEqual(merge_alerts([(0, 3), (1, 4), (7, 10), (8, 11)]), [[0, 4], [7, 11]])
        self.assertEqual(merge_alerts([]), [])

    def test_negative_rates_have_correct_denominators(self):
        clips = [{'flagged': 1}, {'flagged': 0}, {'flagged': 0}, {'flagged': 0}]
        recordings = [{'duration_seconds': 1800, 'alert_events': 3, 'alerted_seconds': 9, 'flagged': 1}]
        r = negative_summary(clips, recordings)
        self.assertEqual(r['candidate_false_alarm_clip_rate'], .25)
        self.assertEqual(r['negative_clip_rejection_rate'], .75)
        self.assertEqual(r['candidate_alert_events_per_hour'], 6.)
        self.assertEqual(r['correctly_rejected_clips_assuming_negative_labels'], 3)


if __name__ == '__main__':
    unittest.main()
