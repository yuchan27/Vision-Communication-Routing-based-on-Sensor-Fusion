import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def choose_input_video() -> Path:
	"""Let user choose a video file via dialog, with console fallback."""
	try:
		import tkinter as tk
		from tkinter import filedialog

		root = tk.Tk()
		root.withdraw()
		selected = filedialog.askopenfilename(
			title="Select a video file",
			filetypes=[
				(
					"Video files",
					"*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.mpeg *.mpg *.ts",
				),
				("All files", "*.*"),
			],
		)
		root.destroy()

		if selected:
			return Path(selected)
	except Exception:
		# If GUI is unavailable, fallback to console input.
		pass

	manual_path = input("Please enter video file path: ").strip().strip('"')
	if not manual_path:
		raise ValueError("No input video selected.")
	return Path(manual_path)


def split_video(
	input_path: Path,
	output_dir: Path,
	segment_seconds: float = 5.0,
	output_extension: str | None = None,
	mode: str = "accurate",
) -> None:
	"""Split a video into roughly fixed-length segments using ffmpeg.

	Notes:
	- Supports any input format that ffmpeg can decode.
	- accurate mode re-encodes for near-exact segment length and better compatibility.
	- fast mode uses stream copy; segment boundaries depend on keyframes.
	"""
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	if segment_seconds <= 0:
		raise ValueError("segment_seconds must be greater than 0")

	if mode not in {"accurate", "fast"}:
		raise ValueError("mode must be 'accurate' or 'fast'")

	if shutil.which("ffmpeg") is None:
		raise RuntimeError(
			"ffmpeg not found. Please install ffmpeg and ensure it is in PATH."
		)

	output_dir.mkdir(parents=True, exist_ok=True)

	ext = output_extension or ".mp4"
	if not ext:
		ext = ".mp4"
	if not ext.startswith("."):
		ext = f".{ext}"

	output_pattern = output_dir / f"{input_path.stem}_part_%04d{ext}"

	command = ["ffmpeg", "-hide_banner", "-y", "-i", str(input_path)]

	if mode == "accurate":
		# Re-encode with forced keyframes for near-exact split durations.
		command.extend(
			[
				"-map",
				"0:v:0",
				"-map",
				"0:a:0?",
				"-c:v",
				"libx264",
				"-preset",
				"veryfast",
				"-crf",
				"23",
				"-c:a",
				"aac",
				"-b:a",
				"128k",
				"-force_key_frames",
				f"expr:gte(t,n_forced*{segment_seconds})",
			]
		)
	else:
		command.extend(["-map", "0", "-c", "copy"])

	command.extend(
		[
			"-f",
			"segment",
			"-segment_time",
			str(segment_seconds),
			"-reset_timestamps",
			"1",
			str(output_pattern),
		]
	)

	result = subprocess.run(command, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError(
			"ffmpeg split failed.\n"
			f"Command: {' '.join(command)}\n"
			f"stderr:\n{result.stderr.strip()}"
		)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Split a video into segments (default ~5 seconds each). "
			"Works with any format supported by ffmpeg."
		)
	)
	parser.add_argument(
		"input",
		nargs="?",
		type=Path,
		help="Input video file path. If omitted, a file picker opens.",
	)
	parser.add_argument(
		"-o",
		"--output-dir",
		type=Path,
		default=Path("dataset"),
		help="Output folder for split files (default: ./dataset)",
	)
	parser.add_argument(
		"-s",
		"--segment-seconds",
		type=float,
		default=5.0,
		help="Length of each segment in seconds (default: 5)",
	)
	parser.add_argument(
		"-e",
		"--output-extension",
		default=None,
		help=(
			"Output file extension, for example mp4 or .mp4. "
			"Default: .mp4"
		),
	)
	parser.add_argument(
		"-m",
		"--mode",
		choices=["accurate", "fast"],
		default="accurate",
		help=(
			"Split mode: accurate (re-encode, near-exact duration, default) or "
			"fast (copy stream, faster but less precise)."
		),
	)
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()
	input_path = args.input or choose_input_video()

	try:
		split_video(
			input_path=input_path,
			output_dir=args.output_dir,
			segment_seconds=args.segment_seconds,
			output_extension=args.output_extension,
			mode=args.mode,
		)
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1

	print(f"Input video: {input_path.resolve()}")
	print(
		f"Done. Segments created in: {args.output_dir.resolve()} "
		f"(~{args.segment_seconds}s each, mode={args.mode})"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
