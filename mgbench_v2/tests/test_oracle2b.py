import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mgbench_v2.oracle2b import Excerpt, Frame, Note

def mk(frame, seq):
    """seq: (onset_in_beats_from_music_start, dur_beats, pitch[, velocity]) -> Excerpt"""
    spb = frame.sec_per_beat
    notes = [Note(frame.music_start + o * spb, frame.music_start + (o + d) * spb, p, v[0] if v else 80) for o, d, p, *v in seq]
    return Excerpt(notes, frame)

F44 = Frame("4/4", 120, 0.5, 2.0)              # bar = 4 beats * 0.5 s = 2 s
F24 = Frame("2/4", 72, 0.4, 2 * 60 / 72)

class T(unittest.TestCase):
    def test_bar_membership_and_frame(self):
        e = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, .5, 62), (5.5, .5, 64), (6, 2, 65)])
        self.assertEqual(e.bars[1], [0, 1, 2]); self.assertEqual(e.bars[2], [3, 4, 5, 6])
        self.assertTrue(e.frame_clear); self.assertTrue(e.usable)
        self.assertEqual(e.bar_initial_notes(), [0, 3])

    def test_pickup_is_not_frame_clear(self):
        e = mk(F44, [(0.5, 1, 60), (1.5, 1, 62), (4, 1, 60), (5, 1, 62)])
        self.assertFalse(e.frame_clear); self.assertIsNone(e.bar_initial_notes()); self.assertIsNone(e.notes_on_beat(1))

    def test_ultra_short_and_overflow_are_unusable(self):
        e = mk(F24, [(0, .5, 60), (.5, .5, 62), (1, .5, 64), (1.5, .5, 65), (2, .06, 66), (2, .5, 67)])
        self.assertFalse(e.usable)

    def test_beat_positions_and_second_half(self):
        e = mk(F44, [(0, 1, 60), (1, .5, 62), (1.5, .5, 64), (2, 1, 65), (3, 1, 67), (4, 2, 60), (6, 2, 62)])
        self.assertEqual(e.notes_on_beat(1), [0, 5]); self.assertEqual(e.notes_on_beat(2), [1])
        self.assertEqual(e.notes_at_bar_middle(), [3, 6]); self.assertEqual(e.notes_in_second_half(bar=1), [3, 4])
        self.assertEqual(e.beat_of_note(2), "the second half of beat 2")

    def test_note_values_and_syncopation(self):
        e = mk(F44, [(0, .5, 60), (.5, 1, 62), (1.5, .5, 64), (2, 2, 65), (4, 1, 60), (5, 1, 62), (6, 2, 64)])
        self.assertEqual(e.value_name(0), "eighth note"); self.assertEqual(e.value_name(3), "half note")
        self.assertEqual(e.syncopated_notes(), [1])           # off-beat start, held across beat 2 -> 3
        self.assertEqual(e.shortest_notes_per_bar(), 8)

    def test_six_eight_uses_pulse_names(self):
        f = Frame("6/8", 90, 0.3, 6 * 60 / 90)
        e = mk(f, [(0, 3, 60), (3, 1, 62), (4, 2, 64), (6, 1, 60), (7, 2, 62), (9, 3, 64)])
        self.assertEqual(e.value_name(0), "dotted quarter note"); self.assertEqual(e.value_name(1), "eighth note")

    def test_tonality(self):
        e = mk(F44, [(0, 1, 60), (1, 1, 61), (2, 1, 64), (3, 1, 67), (4, 1, 60), (5, 1, 62), (6, 1, 66), (7, 1, 67)])
        self.assertEqual(e.notes_outside_scale(0), [1, 6]); self.assertEqual(e.scale_notes_never_played(0), [5, 9, 11])
        self.assertEqual(e.distinct_pitch_classes(), [0, 1, 2, 4, 6, 7])

    def test_intervals(self):
        e = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 1, 62), (3, 1, 68), (4, 1, 65), (5, 1, 66), (6, 1, 60), (7, 1, 72)])
        self.assertEqual(e.notes_reached_by(1, 2), [1, 5])
        self.assertEqual(e.notes_reached_by(3, 5), [4]); self.assertEqual(e.notes_reached_by(6, None), [3, 6, 7])
        self.assertEqual(e.repeated_notes(), [2]); self.assertEqual(e.largest_jump(), 12)
        self.assertEqual(e.notes_reached_by_named(6, "descending"), [6])
        self.assertEqual(e.pitch_rank_notes(2, True), [3]); self.assertEqual(e.pitch_rank_notes(1, False), [0, 6])

    def test_harmony_and_uniqueness(self):
        # bar1 arpeggio of A minor (A C E), bar2 of F major (F A C)
        e = mk(F44, [(0, 1, 57), (1, 1, 60), (2, 1, 64), (3, 1, 57), (4, 1, 53), (5, 1, 57), (6, 1, 60), (7, 1, 53)])
        self.assertEqual(e.triad_of_bar(1), (9, "minor")); self.assertEqual(e.triad_of_bar(2), (5, "major"))
        self.assertEqual(e.triad_name(e.triad_of_bar(1)), "A minor"); self.assertEqual(e.root_motion(), 8)
        self.assertEqual(e.notes_not_in_other_triad(2, 1), [4, 7])       # F is not in A minor
        self.assertEqual(e.triad_member_notes(1, "root"), [0, 3])
        # two notes only -> not a triad
        e2 = mk(F44, [(0, 1, 57), (1, 1, 60), (2, 1, 57), (3, 1, 60), (4, 1, 57), (5, 1, 60), (6, 1, 57), (7, 1, 60)])
        self.assertIsNone(e2.triad_of_bar(1))

    def test_repetition(self):
        e = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)])
        self.assertEqual(e.bar_shift(), (0, 0)); self.assertEqual(e.bar2_notes_differing(), [])
        e = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 65), (5, 1, 67), (6, 2, 69)])
        self.assertEqual(e.bar_shift(), (5, 3))
        e = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 65), (5, 1, 66), (6, 2, 69)])
        self.assertEqual(e.bar_shift(), (None, 3)); self.assertEqual(e.bar2_notes_differing(), [3, 4, 5])

    def test_arpeggio_drops(self):
        asc = mk(F44, [(0, .5, 57), (.5, .5, 60), (1, .5, 64), (1.5, .5, 57), (2, .5, 60), (2.5, .5, 64), (4, .5, 57), (4.5, .5, 60), (5, .5, 64), (5.5, .5, 57)])
        self.assertEqual(asc.drops_to_lowest(1), 1)
        brk = mk(F44, [(0, .5, 57), (.5, .5, 60), (1, .5, 64), (1.5, .5, 60), (2, .5, 57), (2.5, .5, 60), (4, .5, 57), (4.5, .5, 60), (5, .5, 64), (5.5, .5, 60)])
        self.assertEqual(brk.downward_notes(), [3, 4, 6, 9])   # 6 = bar-boundary move 60 -> 57; self.assertEqual(brk.drops_to_lowest(1), 1)

    def test_tempo_and_beats_per_bar_need_consistent_bar_length(self):
        e = mk(F44, [(0, 1, 60), (1, 1, 62), (4, 1, 60), (5, 1, 62)])
        self.assertEqual(e.tempo_answer(), 120); self.assertEqual(e.beats_per_bar_answer(), 4)

    def test_longest_note_margin(self):
        e = mk(F44, [(0, 2, 60), (2, 1.5, 62), (3.5, .5, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)])
        self.assertEqual(e.longest_note_of_bar(1, 1.3), 0)        # 2.0 / 1.5 = 1.33 >= 1.3
        self.assertIsNone(e.longest_note_of_bar(1, 1.5))          # ... but not >= 1.5
        self.assertEqual(e.longest_note_of_bar(2, 1.3), 5)        # bar 2: the last note counts (2 beats, up to the end of bar 2) and clearly beats notes 3, 4 (1 beat)

    def test_last_note_counts_up_to_the_end_of_bar_2(self):
        e = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)])
        self.assertEqual(e.value_name(5), "half note"); self.assertEqual(e.notes_with_value(2.0), [2, 5])
        self.assertEqual(e.syncopated_notes(), [])

    def test_rests_switch_off_length_questions(self):
        rest = mk(F44, [(0, 1, 60), (1, .5, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)])          # note 1 sounds 0.5 beat, the next onset is 1 beat later
        self.assertFalse(rest.rest_free); self.assertFalse(rest.values_ok)
        self.assertIsNone(rest.notes_with_value(1.0)); self.assertIsNone(rest.syncopated_notes()); self.assertIsNone(rest.longest_note_of_bar(1, 1.3))
        early = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 1, 64)])          # the last note stops 1 beat before the end of bar 2
        self.assertFalse(early.rest_free); self.assertIsNone(early.notes_with_value(1.0))
        self.assertTrue(early.frame_clear and early.grid_ok)                                                 # ... beat questions are still fine
        tiny = mk(F44, [(0, .95, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)])         # a 25 ms articulation gap is not a rest
        self.assertTrue(tiny.rest_free)

    def test_syncopation_needs_a_clear_crossing(self):
        base = [(0, .5, 60), (.5, 1, 62), (1.5, .5, 64), (2, 2, 65), (4, 1, 60), (5, 1, 62), (6, 2, 64)]
        self.assertEqual(mk(F44, base).syncopated_notes(), [1])
        # an off-beat note that reaches only 0.15 beat past the stronger position is neither "held through" it nor "not held"
        blurred = [(0, .5, 60), (.5, .65, 62), (1.15, .85, 64), (2, 2, 65), (4, 1, 60), (5, 1, 62), (6, 2, 64)]
        self.assertIsNone(mk(F44, blurred).syncopated_notes())

    def test_syncopation_is_weaker_to_stronger_position(self):
        # 4/4: a dotted quarter on beat 2 is held through beat 3 (stronger) -> syncopated, although it does not start "off the beat"
        weak_beat = [(0, 1, 60), (1, 1.5, 62), (2.5, .5, 64), (3, 1, 65), (4, 1, 60), (5, 1, 62), (6, 2, 64)]
        self.assertEqual(mk(F44, weak_beat).syncopated_notes(), [1])
        # 4/4: a half note on beat 3 (strongest but the downbeat) reaching the bar line is not syncopated; nor is a quarter on beat 2 that ends on beat 3
        plain = [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)]
        self.assertEqual(mk(F44, plain).syncopated_notes(), [])
        # 4/4: an off-beat quarter note between two eighths (the textbook case) is held through beat 3
        textbook = [(0, 1, 60), (1, 1, 62), (2, .5, 64), (2.5, 1, 65), (3.5, .5, 60), (4, 1, 62), (5, 1, 64), (6, 2, 65)]
        self.assertEqual(mk(F44, textbook).syncopated_notes(), [3])
        # 3/4: beats 2 and 3 are equally strong: a half note on beat 2 is not syncopated
        F34 = Frame("3/4", 120, 0.5, 1.5)
        self.assertEqual(mk(F34, [(0, 1, 60), (1, 2, 62), (3, 1, 60), (4, 2, 62)]).syncopated_notes(), [])

    def test_no_syncopation_question_in_6_8(self):
        F68 = Frame("6/8", 120, 0.5, 3.0)
        e = mk(F68, [(0, 1.5, 60), (1.5, 1.5, 62), (3, 1.5, 60), (4.5, 1.5, 62), (6, 1.5, 60), (7.5, 1.5, 62), (9, 1.5, 60), (10.5, 1.5, 62)])
        self.assertTrue(e.values_ok); self.assertIsNone(e.syncopated_notes())

    def test_arpeggio_shape(self):
        # 4/4, two bars of eight notes: bar 1 ascending (low mid high | low mid high | low mid), bar 2 up-and-down (low mid high mid | low mid high mid)
        seq = [(i * .5, .5, p) for i, p in enumerate([60, 64, 67, 60, 64, 67, 60, 64])] + [(4 + i * .5, .5, p) for i, p in enumerate([62, 65, 69, 65, 62, 65, 69, 65])]
        e = mk(F44, seq)
        self.assertEqual(e.arpeggio_kind(1), "ascending"); self.assertEqual(e.arpeggio_kind(2), "updown")
        self.assertEqual(e.arpeggio_first_cycle(1), [0, 1, 2]); self.assertEqual(e.arpeggio_first_cycle(2), [8, 9, 10, 11])
        self.assertEqual(e.drop_back_notes(1), [3, 6]); self.assertEqual(e.drop_back_notes(2), [12])     # up-and-down: only the bar's low note after the mid note counts
        # a bar of three notes cannot tell the shapes apart; a bar with a fourth pitch has no shape
        e3 = mk(F44, [(0, 1, 60), (1, 1, 64), (2, 2, 67), (4, 1, 62), (5, 1, 65), (6, 2, 69)])
        self.assertIsNone(e3.arpeggio_kind(1))
        e4 = mk(F44, [(0, 1, 60), (1, 1, 64), (2, 1, 67), (3, 1, 72), (4, 1, 62), (5, 1, 65), (6, 1, 69), (7, 1, 65)])
        self.assertIsNone(e4.arpeggio_kind(1))

    def test_frame_full_needs_first_note_at_the_start_and_last_note_at_the_end(self):
        ok = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)])
        self.assertTrue(ok.frame_full)
        early = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 1, 64)])       # last note stops one beat before the end of bar 2
        self.assertTrue(early.frame_clear); self.assertFalse(early.frame_full)

    def test_bar_ambiguous(self):
        e = mk(F44, [(0, 1, 60), (1, 1, 62), (3.7, .3, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)])   # a note 0.3 beat = 0.15 s before the bar line
        self.assertTrue(e.bar_ambiguous)
        e2 = mk(F44, [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 2, 64)])
        self.assertFalse(e2.bar_ambiguous)
