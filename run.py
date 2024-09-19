import argparse
import os
import subprocess


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_argument():
    parser = argparse.ArgumentParser(description="""Recognize all images in folder""")
    parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        default="test_data/ToRecognize",
        help="directories of images",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="test_data/Recognized",
        help="output directory",
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        default=1,
        help="the maximum number of concurrently running jobs",
    )
    parser.add_argument(
        "-lidx", "--last_idx", type=int, default=None, help="last idx. don't use it!"
    )
    parser.add_argument(
        "-sb",
        "--shared_bottom",
        type=str2bool,
        nargs="?",
        const="True",
        default=True,
        help="use shared_bottom or not. default True (use)",
    )
    parser.add_argument(
        "-ip",
        "--image_preprocessing",
        type=str2bool,
        nargs="?",
        const="True",
        default=False,
        help="image preprocessing flag",
    )
    return parser.parse_args()


def main():
    """sh prepare.sh && python run.py \
-i test_data/ToRecognize \
-o test_data/Recognized \
-j 4 \
-sb true \
-ip true
"""
    args = parse_argument()
    os.makedirs(args.out_dir, exist_ok=True)

    docker_in_cmd = f"""cd /workdir; python run_inner.py -i /ToRecognize -o /Recognized -j {args.n_jobs} -sb {args.shared_bottom} -ip {args.image_preprocessing}"""
    if args.last_idx is not None:
        docker_in_cmd += ' -lidx {args.last_idx}'
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--mount",
        f"type=bind,source={os.path.abspath(args.in_dir)},target=/ToRecognize",
        "--mount",
        f"type=bind,source={os.path.abspath(args.out_dir)},target=/Recognized",
        "nekrasovka_recognition:v5.0.1",
        "/bin/bash",
        "-c",
        docker_in_cmd,
    ]
    print('Run cmd:', ' '.join(docker_cmd))
    subprocess.run(docker_cmd, check=True)


if __name__ == "__main__":
    main()
