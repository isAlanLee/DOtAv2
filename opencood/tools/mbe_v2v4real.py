"""
Compatibility wrapper for the V2V4Real MBE workflow.

Use the two-stage scripts for paper-aligned reproduction:
1. opencood/tools/v2v4real_mbe_filter.py
2. opencood/tools/v2v4real_box_score.py
"""


def main():
    raise SystemExit(
        'mbe_v2v4real.py has been replaced by a two-stage workflow. '
        'Run opencood/tools/v2v4real_mbe_filter.py first, then '
        'opencood/tools/v2v4real_box_score.py.'
    )


if __name__ == '__main__':
    main()
