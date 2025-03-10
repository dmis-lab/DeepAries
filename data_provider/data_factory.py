
from data_provider.data_loader import TimeSeriesDataset
from torch.utils.data import DataLoader


def data_provider(args, flag):
    """
    Data provider function to return the dataset and dataloader for a given flag.

    Args:
        args: Argument parser object containing parameters.
        flag: Data split flag ('train', 'val', 'test', 'backtest', 'pred').

    Returns:
        tuple: (dataset, DataLoader)
    """
    if flag in ['test', 'backtest']:
        shuffle_flag = False
        drop_last = (flag == 'test')
        batch_size = 1
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        # For training and validation, always shuffle.
        shuffle_flag = True
        drop_last = True
        batch_size = getattr(args, 'batch_size', 1)

    # For DeepAries, always use multi-horizon labels with step sampling disabled.
    use_multi_horizon = True
    use_step_sampling = False
    lookaheads = args.horizons

    dataset = TimeSeriesDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        data=args.data,
        flag=flag,
        valid_year=args.valid_year,
        test_year=args.test_year,
        size=[args.seq_len, args.label_len, args.pred_len],
        use_multi_horizon=use_multi_horizon,
        lookaheads=lookaheads,
        scale=True,
        timeenc=(1 if args.embed == 'timeF' else 0),
        freq=args.freq,
        step_size=args.pred_len,
        use_step_sampling=use_step_sampling
    )

    print(f"{flag} dataset length: {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        persistent_workers=True
    )
    return dataset, data_loader
