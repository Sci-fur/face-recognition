#!/usr/bin/env python3
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Real-Time Face Recognition System — Computer Peripheral Lab'
    )
    sub = parser.add_subparsers(dest='command', help='Available commands')

    p_collect = sub.add_parser('collect', help='Collect face images for training')
    p_collect.add_argument('--name', '-n', required=True,
                           help='Name of the person to collect data for')

    sub.add_parser('train', help='Train the face recognition model')

    sub.add_parser('recognize', help='Start real-time face recognition')

    args = parser.parse_args()

    if args.command == 'collect':
        from collect import collect
        collect(args.name)
    elif args.command == 'train':
        from train import train_model
        train_model()
    elif args.command == 'recognize':
        from recognize import recognize
        recognize()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
