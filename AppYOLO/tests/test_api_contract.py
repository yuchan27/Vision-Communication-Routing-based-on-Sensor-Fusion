import unittest

from src.api_contract import (
    VIDEO_SUFFIXES,
    build_temperature_observation,
    format_sse,
    parse_optional_temperature,
    validate_upload_filename,
    validate_upload_size,
)


class UploadContractTests(unittest.TestCase):
    def test_video_suffixes_are_case_insensitive_and_include_m4v(self):
        self.assertEqual(validate_upload_filename("clip.M4V", VIDEO_SUFFIXES, "video"), "clip.M4V")

    def test_unsupported_upload_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_upload_filename("payload.gif", {".jpg"}, "image")

    def test_upload_limit_is_inclusive(self):
        validate_upload_size(10, 10, "image")
        with self.assertRaises(ValueError):
            validate_upload_size(11, 10, "image")


class TemperatureContractTests(unittest.TestCase):
    def test_sensor_temperature_parser_rejects_non_finite_or_out_of_range_values(self):
        self.assertEqual(parse_optional_temperature(""), None)
        with self.assertRaises(ValueError):
            parse_optional_temperature("nan")
        with self.assertRaises(ValueError):
            parse_optional_temperature(201)

    def test_sensor_temperature_is_preferred_and_marked_calibrated(self):
        observation = build_temperature_observation(
            sensor_temperature_celsius=83.4,
            rgb_temperature_celsius=410.0,
        )

        self.assertEqual(observation["scene_temperature_celsius"], 83.4)
        self.assertEqual(observation["scene_temperature_source"], "thermal_sensor")
        self.assertTrue(observation["scene_temperature_calibrated"])

    def test_rgb_temperature_is_explicitly_an_estimate(self):
        observation = build_temperature_observation(
            sensor_temperature_celsius=None,
            rgb_temperature_celsius=410.0,
        )

        self.assertEqual(observation["scene_temperature_celsius"], 410.0)
        self.assertEqual(observation["scene_temperature_source"], "rgb_estimate")
        self.assertFalse(observation["scene_temperature_calibrated"])


class SseContractTests(unittest.TestCase):
    def test_sse_uses_real_record_separators(self):
        event = format_sse({"frame_id": 7})

        self.assertTrue(event.endswith("\n\n"))
        self.assertNotIn("\\n\\n", event)


if __name__ == "__main__":
    unittest.main()
