from src.load_data import load_train


def test_no_null_loss(tmp_path):
    # a malformed row with null loss should be dropped
    csv = tmp_path / "train.csv"
    csv.write_text("id,cat1,cont1,loss\n1,A,0.1,100.0\n2,B,0.2,\n3,A,0.3,300.0\n")

    df = load_train(str(csv))

    assert df["loss"].isnull().sum() == 0
    assert len(df) == 2
