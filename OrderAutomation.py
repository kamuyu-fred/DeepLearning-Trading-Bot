"""GUI execution is disabled pending verified position and protection handling."""


def main():
    raise SystemExit(
        'GUI order placement is disabled: OCR and fixed coordinates cannot reliably '
        'identify a position or verify its stop-loss and take-profit. '
        'Use the MT5 API dry-run workflow instead.'
    )


if __name__ == '__main__':
    main()
