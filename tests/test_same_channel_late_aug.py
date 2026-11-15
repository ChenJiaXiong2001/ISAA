import unittest

import train_best_weighted_100_same_channel_late_aug as entry


class SameChannelLateAugTests(unittest.TestCase):
    def test_defaults(self):
        args = entry.training_arguments([])
        self.assertIn("--model-variant", args)
        self.assertEqual(args[args.index("--model-variant") + 1],
                         "body-local-hand-ctr-wide-relative-same-channel-late-aug")
        self.assertEqual(args[args.index("--epochs") + 1], "100")
        self.assertEqual(args[args.index("--late-augmentation-start") + 1], "55")

    def test_rejects_no_compile(self):
        with self.assertRaises(ValueError):
            entry.training_arguments(["--no-compile"])


if __name__ == "__main__":
    unittest.main()
