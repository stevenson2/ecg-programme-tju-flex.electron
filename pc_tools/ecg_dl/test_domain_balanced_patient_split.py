from data.dataset import prepare_datasets


def test_domain_balanced_patient_split_first_batch_has_matching_dtypes():
    datasets = prepare_datasets(
        batch_size=128,
        use_incart=True,
        use_ptb_beat=True,
        ptb_abn_max=10000,
        domain_balanced=True,
        patient_split=True,
    )
    next(iter(datasets["train_ds"]))


if __name__ == "__main__":
    test_domain_balanced_patient_split_first_batch_has_matching_dtypes()
    print("PASS: domain-balanced patient split first batch")
